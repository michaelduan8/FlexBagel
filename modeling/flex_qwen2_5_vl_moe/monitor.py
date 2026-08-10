from collections import defaultdict
import torch
import torch.nn.functional as F


class MoeRoutingMonitor:
    """
    MoE routing monitor for k=1 inference.

    Tracks:
      - expert usage
      - selected probability / entropy
      - expert switch rate along routing paths
      - expert transition matrix
      - mean same-expert run length
      - micro-averaged language / vision summaries

    Perf notes vs. the naive version:
      - All per-call accumulation stays on-device as tensors. Nothing calls
        .item() / .cpu() / .tolist() (or branches on a device tensor's
        truthiness) inside the hook. Those force a host<->device sync,
        which stalls the CUDA stream on every single hooked forward call.
        We only sync once, lazily, when summary() is actually called.
      - The old per-timestep Python `for t in range(seq_len)` loop is gone.
        Transitions, switch/stay counts, and same-expert run lengths are
        all computed with a handful of vectorized ops (cat/compare/cummax)
        over the whole [batch, seq_len] chunk at once.
      - The old per-batch-row `torch.unique()` Python loop is replaced with
        a sort + adjacent-diff trick, vectorized across the batch.
      - probs.max(dim=-1) is computed once instead of twice.
    """

    def __init__(
        self,
        model,
        skip_first_call_per_layer: bool = False,
        track_transition_matrix: bool = True,
    ):
        self.model = model
        self.skip_first_call_per_layer = skip_first_call_per_layer
        self.track_transition_matrix = track_transition_matrix
        self.handles = []
        self.stats = {}
        self.call_count = defaultdict(int)

        # For path continuity across generate() calls.
        # layer_name -> Tensor[batch] (kept on the same device as that layer's router logits)
        self.last_expert = {}
        self.active_run_len = {}

    def _layer_group(self, layer_name: str):
        if "language_model.layers." in layer_name and layer_name.endswith(".mlp"):
            return "language_mlp"
        if "visual.blocks." in layer_name and layer_name.endswith(".mlp"):
            return "vision_mlp"
        return None

    def _new_layer_stats(self, num_experts: int, device: torch.device):
        return {
            "tokens": 0,
            "assignments": 0,
            "num_calls": 0,

            # Usage. Stay on-device; only moved to CPU in summary().
            "top1_counts": torch.zeros(num_experts, dtype=torch.long, device=device),
            "topk_counts": torch.zeros(num_experts, dtype=torch.long, device=device),
            "weight_sums": torch.zeros(num_experts, dtype=torch.float32, device=device),

            # Confidence. 0-d device tensors instead of python floats to avoid
            # a .item() sync on every call.
            "entropy_sum_t": torch.zeros((), dtype=torch.float32, device=device),
            "max_prob_sum_t": torch.zeros((), dtype=torch.float32, device=device),

            # Path stats.
            "transition_counts": (
                torch.zeros(num_experts, num_experts, dtype=torch.long, device=device)
                if self.track_transition_matrix
                else None
            ),
            "transition_count_t": torch.zeros((), dtype=torch.long, device=device),
            "switch_count_t": torch.zeros((), dtype=torch.long, device=device),
            "run_count_t": torch.zeros((), dtype=torch.long, device=device),
            "max_run_len_t": torch.zeros((), dtype=torch.long, device=device),
            "unique_experts_sum_t": torch.zeros((), dtype=torch.long, device=device),
            "num_paths_observed": 0,  # plain int: += batch_size, never touches the device
        }

    def _reset_path_state_for_layer(self, layer_name: str, batch_size: int, device: torch.device):
        self.last_expert[layer_name] = torch.full(
            (batch_size,), -1, dtype=torch.long, device=device,
        )
        self.active_run_len[layer_name] = torch.zeros(
            batch_size, dtype=torch.long, device=device,
        )

    def _update_path_stats(self, layer_name: str, expert_path: torch.Tensor, s: dict):
        """
        expert_path: LongTensor [batch, seq_len], on the same device as router_logits.
        Fully vectorized - no python-level loop over seq_len or batch.
        """
        batch_size, seq_len = expert_path.shape
        if seq_len == 0:
            return

        device = expert_path.device

        if (
            layer_name not in self.last_expert
            or self.last_expert[layer_name].numel() != batch_size
            or self.last_expert[layer_name].device != device
        ):
            self._reset_path_state_for_layer(layer_name, batch_size, device)

        last = self.last_expert[layer_name]                # [batch]
        active_run_len = self.active_run_len[layer_name]    # [batch]

        # ---- unique experts per path segment (sort + adjacent-diff, no python loop) ----
        sorted_path, _ = torch.sort(expert_path, dim=1)
        unique_counts = (sorted_path[:, 1:] != sorted_path[:, :-1]).sum(dim=1) + 1
        s["unique_experts_sum_t"] += unique_counts.sum()
        s["num_paths_observed"] += batch_size

        # ---- prev/cur pairs for the whole chunk at once ----
        full = torch.cat([last.unsqueeze(1), expert_path], dim=1)  # [batch, seq_len+1]
        prev = full[:, :-1]
        cur = full[:, 1:]
        valid_prev = prev >= 0  # False only where `last` was the -1 sentinel

        # NOTE: intentionally no `if valid_prev.any():` guard here - that would force
        # a host sync every call. Boolean-masked ops on an all-False mask are cheap
        # no-ops on an empty tensor, so we just always run them.
        prev_valid = prev[valid_prev]
        cur_valid = cur[valid_prev]

        if s["transition_counts"] is not None:
            num_experts = s["transition_counts"].shape[0]
            flat_trans = prev_valid * num_experts + cur_valid
            trans_counts = torch.bincount(
                flat_trans, minlength=num_experts * num_experts,
            ).reshape(num_experts, num_experts)
            s["transition_counts"] += trans_counts

        s["transition_count_t"] += valid_prev.sum()
        s["switch_count_t"] += (prev_valid != cur_valid).sum()

        # ---- run-length tracking, vectorized via cummax instead of a python loop ----
        start_new_run = (~valid_prev) | (prev != cur)  # [batch, seq_len]
        s["run_count_t"] += start_new_run.sum()

        seq_idx = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, seq_len)
        reset_idx = torch.where(start_new_run, seq_idx, torch.full_like(seq_idx, -1))
        last_reset_idx, _ = torch.cummax(reset_idx, dim=1)
        no_reset_yet = last_reset_idx < 0  # run continues from before this chunk

        continued_run_len = active_run_len.unsqueeze(1) + seq_idx + 1
        post_reset_run_len = seq_idx - last_reset_idx + 1
        run_len = torch.where(no_reset_yet, continued_run_len, post_reset_run_len)

        s["max_run_len_t"] = torch.maximum(s["max_run_len_t"], run_len.max())

        # carry state forward for the next hooked call (e.g. next decode step)
        active_run_len.copy_(run_len[:, -1])
        last.copy_(cur[:, -1])

    def _make_hook(self, layer_name, moe_block):
        def hook(module, inputs, output):
            # SparseMoeBlock returns: (final_hidden_states, router_logits)
            if not isinstance(output, tuple) or len(output) < 2:
                return

            router_logits = output[1]
            if router_logits is None or not torch.is_tensor(router_logits):
                return

            self.call_count[layer_name] += 1
            if self.skip_first_call_per_layer and self.call_count[layer_name] == 1:
                return

            hidden_states_in = inputs[0]
            if hidden_states_in.ndim == 2:
                batch_size = 1
                seq_len = hidden_states_in.shape[0]
            elif hidden_states_in.ndim == 3:
                batch_size = hidden_states_in.shape[0]
                seq_len = hidden_states_in.shape[1]
            else:
                return

            with torch.inference_mode():
                router_logits = router_logits.detach()
                probs = F.softmax(router_logits.float(), dim=-1)

                num_experts = moe_block.num_experts

                # k=1, so this is the actual routed expert. Computed once and
                # reused (the old code called probs.max(dim=-1) twice).
                max_prob, selected_expert = probs.max(dim=-1)

                num_tokens = selected_expert.numel()
                expert_path = selected_expert.reshape(batch_size, seq_len)

                if layer_name not in self.stats:
                    self.stats[layer_name] = self._new_layer_stats(
                        num_experts, router_logits.device,
                    )

                s = self.stats[layer_name]

                # bincount runs directly on-device; no .cpu() needed.
                top1_counts = torch.bincount(selected_expert, minlength=num_experts)

                weight_sums = torch.zeros(
                    num_experts, dtype=torch.float32, device=router_logits.device,
                )
                weight_sums.scatter_add_(0, selected_expert, max_prob.float())

                entropy = -(probs * probs.clamp_min(1e-20).log()).sum(dim=-1)

                s["tokens"] += num_tokens
                s["assignments"] += num_tokens
                s["num_calls"] += 1

                # For k=1, top-k counts are identical to top-1 counts.
                s["top1_counts"] += top1_counts
                s["topk_counts"] += top1_counts
                s["weight_sums"] += weight_sums

                s["entropy_sum_t"] += entropy.sum()
                s["max_prob_sum_t"] += max_prob.sum()

                self._update_path_stats(layer_name, expert_path, s)

        return hook

    def register(self):
        self.remove()

        print("[MoeRoutingMonitor] MoE blocks found:")
        for name, module in self.model.named_modules():
            is_moe_block = (
                hasattr(module, "gate")
                and hasattr(module, "num_experts")
                and hasattr(module, "top_k")
                and hasattr(module, "experts")
            )

            if is_moe_block:
                print("  ", name)
                handle = module.register_forward_hook(
                    self._make_hook(name, module)
                )
                self.handles.append(handle)

        print(f"[MoeRoutingMonitor] registered {len(self.handles)} MoE block hooks")
        return self

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    def reset(self):
        self.stats = {}
        self.call_count = defaultdict(int)
        self.last_expert = {}
        self.active_run_len = {}

    def summary(self):
        out = {}

        for layer_name, s in self.stats.items():
            tokens = max(s["tokens"], 1)
            assignments = max(s["assignments"], 1)
            num_paths = max(s["num_paths_observed"], 1)

            # Single sync point per layer - everything needed is pulled to host
            # here, once, instead of on every hooked forward call.
            entropy_sum = float(s["entropy_sum_t"].item())
            max_prob_sum = float(s["max_prob_sum_t"].item())
            transition_count = int(s["transition_count_t"].item())
            switch_count = int(s["switch_count_t"].item())
            run_count = int(s["run_count_t"].item())
            max_run_len = int(s["max_run_len_t"].item())
            unique_experts_sum = int(s["unique_experts_sum_t"].item())

            transition_count_safe = max(transition_count, 1)
            run_count_safe = max(run_count, 1)

            top1_counts_cpu = s["top1_counts"].cpu()
            topk_counts_cpu = s["topk_counts"].cpu()
            weight_sums_cpu = s["weight_sums"].cpu()

            top1_frac = top1_counts_cpu.float() / tokens
            topk_frac = topk_counts_cpu.float() / assignments
            avg_selected_weight = weight_sums_cpu / top1_counts_cpu.clamp_min(1).float()

            switch_rate = switch_count / transition_count_safe
            stay_rate = 1.0 - switch_rate
            mean_run_len = s["tokens"] / run_count_safe
            avg_unique_experts_per_path = unique_experts_sum / num_paths

            transition_counts = None
            transition_frac = None
            if s["transition_counts"] is not None:
                transition_counts_cpu = s["transition_counts"].cpu()
                transition_counts = transition_counts_cpu.tolist()
                transition_frac = (
                    transition_counts_cpu.float() / transition_count_safe
                ).tolist()

            out[layer_name] = {
                "num_calls": s["num_calls"],
                "tokens": s["tokens"],
                "assignments": s["assignments"],

                "top1_counts": top1_counts_cpu.tolist(),
                "top1_frac": top1_frac.tolist(),
                "topk_counts": topk_counts_cpu.tolist(),
                "topk_frac": topk_frac.tolist(),
                "avg_selected_weight": avg_selected_weight.tolist(),

                "avg_entropy": entropy_sum / tokens,
                "avg_max_prob": max_prob_sum / tokens,

                "transition_counts": transition_counts,
                "transition_frac": transition_frac,
                "transition_count": transition_count,
                "switch_count": switch_count,
                "switch_rate": switch_rate,
                "stay_rate": stay_rate,
                "mean_run_len": mean_run_len,
                "max_run_len": max_run_len,
                "avg_unique_experts_per_path": avg_unique_experts_per_path,
            }

        return out

    def grouped_summary(self):
        per_layer = self.summary()

        groups = {
            "language_mlp_micro": [],
            "vision_mlp_micro": [],
        }

        for layer_name, s in per_layer.items():
            group = self._layer_group(layer_name)
            if group == "language_mlp":
                groups["language_mlp_micro"].append((layer_name, s))
            elif group == "vision_mlp":
                groups["vision_mlp_micro"].append((layer_name, s))

        out = {}

        for group_name, items in groups.items():
            if len(items) == 0:
                continue

            num_experts = len(items[0][1]["top1_counts"])
            has_transition_matrix = items[0][1]["transition_counts"] is not None

            top1_counts = torch.zeros(num_experts, dtype=torch.long)
            topk_counts = torch.zeros(num_experts, dtype=torch.long)
            weight_sums = torch.zeros(num_experts, dtype=torch.float32)
            transition_counts = (
                torch.zeros(num_experts, num_experts, dtype=torch.long)
                if has_transition_matrix
                else None
            )

            tokens = 0
            assignments = 0
            num_calls = 0
            entropy_token_sum = 0.0
            max_prob_token_sum = 0.0

            transition_count = 0
            switch_count = 0
            run_count = 0
            max_run_len = 0
            unique_experts_sum = 0.0
            num_paths_observed = 0

            for _, s in items:
                layer_top1_counts = torch.tensor(s["top1_counts"], dtype=torch.long)
                layer_topk_counts = torch.tensor(s["topk_counts"], dtype=torch.long)
                layer_avg_weight = torch.tensor(s["avg_selected_weight"], dtype=torch.float32)

                top1_counts += layer_top1_counts
                topk_counts += layer_topk_counts
                weight_sums += layer_avg_weight * layer_top1_counts.float()

                if transition_counts is not None:
                    transition_counts += torch.tensor(s["transition_counts"], dtype=torch.long)

                tokens += s["tokens"]
                assignments += s["assignments"]
                num_calls += s["num_calls"]

                entropy_token_sum += s["avg_entropy"] * s["tokens"]
                max_prob_token_sum += s["avg_max_prob"] * s["tokens"]

                transition_count += s["transition_count"]
                switch_count += s["switch_count"]

                # Approximate micro run count from layer summary:
                # mean_run_len = tokens / run_count
                if s["mean_run_len"] > 0:
                    run_count += int(round(s["tokens"] / s["mean_run_len"]))

                max_run_len = max(max_run_len, s["max_run_len"])

                unique_experts_sum += s["avg_unique_experts_per_path"]
                num_paths_observed += 1

            tokens_safe = max(tokens, 1)
            assignments_safe = max(assignments, 1)
            transition_safe = max(transition_count, 1)
            run_safe = max(run_count, 1)

            transition_frac = None
            transition_counts_list = None
            if transition_counts is not None:
                transition_counts_list = transition_counts.tolist()
                transition_frac = (transition_counts.float() / transition_safe).tolist()

            out[group_name] = {
                "num_layers": len(items),
                "layer_names": [name for name, _ in items],
                "num_calls": num_calls,
                "tokens": tokens,
                "assignments": assignments,

                "top1_counts": top1_counts.tolist(),
                "top1_frac": (top1_counts.float() / tokens_safe).tolist(),
                "topk_counts": topk_counts.tolist(),
                "topk_frac": (topk_counts.float() / assignments_safe).tolist(),
                "avg_selected_weight": (
                    weight_sums / top1_counts.clamp_min(1).float()
                ).tolist(),

                "avg_entropy": entropy_token_sum / tokens_safe,
                "avg_max_prob": max_prob_token_sum / tokens_safe,

                "transition_counts": transition_counts_list,
                "transition_frac": transition_frac,
                "transition_count": transition_count,
                "switch_count": switch_count,
                "switch_rate": switch_count / transition_safe,
                "stay_rate": 1.0 - switch_count / transition_safe,
                "mean_run_len": tokens / run_safe,
                "max_run_len": max_run_len,

                # Average over layers, not token-micro. This one is just diagnostic.
                "avg_unique_experts_per_path": unique_experts_sum / max(num_paths_observed, 1),
            }

        return out

    def print_summary(self, max_layers=None):
        summary = self.summary()
        items = list(summary.items())
        if max_layers is not None:
            items = items[:max_layers]

        for layer_name, s in items:
            print("=" * 100)
            print(f"[MoE routing] {layer_name}")
            print(
                f"calls={s['num_calls']}  "
                f"tokens={s['tokens']}  "
                f"switch_rate={100 * s['switch_rate']:.2f}%  "
                f"stay_rate={100 * s['stay_rate']:.2f}%"
            )
            print(
                f"mean_run_len={s['mean_run_len']:.2f}  "
                f"max_run_len={s['max_run_len']}  "
                f"avg_unique_experts_per_path={s['avg_unique_experts_per_path']:.2f}"
            )
            print(
                f"avg_entropy={s['avg_entropy']:.4f}  "
                f"avg_max_prob={s['avg_max_prob']:.4f}"
            )

            print("expert | count | usage_% | avg_selected_weight")
            for i in range(len(s["top1_counts"])):
                print(
                    f"{i:>6} | "
                    f"{s['top1_counts'][i]:>8} | "
                    f"{100 * s['top1_frac'][i]:>7.2f} | "
                    f"{s['avg_selected_weight'][i]:.4f}"
                )

    def print_grouped_summary(self):
        grouped = self.grouped_summary()

        for group_name, s in grouped.items():
            print("=" * 100)
            print(f"[MoE routing MICRO GROUP] {group_name}")
            print(
                f"layers={s['num_layers']}  "
                f"calls={s['num_calls']}  "
                f"tokens={s['tokens']}  "
                f"switch_rate={100 * s['switch_rate']:.2f}%  "
                f"stay_rate={100 * s['stay_rate']:.2f}%"
            )
            print(
                f"mean_run_len={s['mean_run_len']:.2f}  "
                f"max_run_len={s['max_run_len']}  "
                f"avg_entropy={s['avg_entropy']:.4f}  "
                f"avg_max_prob={s['avg_max_prob']:.4f}"
            )

            print("expert | count | usage_% | avg_selected_weight")
            for i in range(len(s["top1_counts"])):
                print(
                    f"{i:>6} | "
                    f"{s['top1_counts'][i]:>8} | "
                    f"{100 * s['top1_frac'][i]:>7.2f} | "
                    f"{s['avg_selected_weight'][i]:.4f}"
                )

    def print_top_transitions(self, grouped: bool = True, top_n: int = 10):
        data = self.grouped_summary() if grouped else self.summary()

        for name, s in data.items():
            if s["transition_counts"] is None:
                print("=" * 100)
                print(f"[MoE top transitions] {name}")
                print("transition matrix tracking disabled")
                continue

            matrix = torch.tensor(s["transition_counts"], dtype=torch.long)
            num_experts = matrix.shape[0]

            flat = matrix.reshape(-1)
            values, indices = torch.topk(flat, k=min(top_n, flat.numel()))

            print("=" * 100)
            print(f"[MoE top transitions] {name}")
            print("prev -> next | count | percent_of_transitions")

            denom = max(int(matrix.sum().item()), 1)

            for value, index in zip(values.tolist(), indices.tolist()):
                if value == 0:
                    continue
                prev_e = index // num_experts
                next_e = index % num_experts
                print(
                    f"{prev_e:>4} -> {next_e:<4} | "
                    f"{value:>8} | "
                    f"{100 * value / denom:>7.2f}%"
                )