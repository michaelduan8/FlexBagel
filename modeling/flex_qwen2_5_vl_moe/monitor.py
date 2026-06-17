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
    """

    def __init__(self, model, skip_first_call_per_layer: bool = False):
        self.model = model
        self.skip_first_call_per_layer = skip_first_call_per_layer
        self.handles = []
        self.stats = {}
        self.call_count = defaultdict(int)

        # For path continuity across generate() calls.
        # layer_name -> Tensor[batch]
        self.last_expert = {}
        self.active_run_len = {}

    def _layer_group(self, layer_name: str):
        if "language_model.layers." in layer_name and layer_name.endswith(".mlp"):
            return "language_mlp"
        if "visual.blocks." in layer_name and layer_name.endswith(".mlp"):
            return "vision_mlp"
        return None

    def _new_layer_stats(self, num_experts: int):
        return {
            "tokens": 0,
            "assignments": 0,
            "num_calls": 0,

            # Usage.
            "top1_counts": torch.zeros(num_experts, dtype=torch.long),
            "topk_counts": torch.zeros(num_experts, dtype=torch.long),
            "weight_sums": torch.zeros(num_experts, dtype=torch.float32),

            # Confidence.
            "entropy_sum": 0.0,
            "max_prob_sum": 0.0,

            # Path stats.
            "transition_counts": torch.zeros(num_experts, num_experts, dtype=torch.long),
            "transition_count": 0,
            "switch_count": 0,
            "run_count": 0,
            "max_run_len": 0,
            "unique_experts_sum": 0,
            "num_paths_observed": 0,
        }

    def _reset_path_state_for_layer(self, layer_name: str, batch_size: int):
        self.last_expert[layer_name] = torch.full(
            (batch_size,),
            -1,
            dtype=torch.long,
        )
        self.active_run_len[layer_name] = torch.zeros(
            batch_size,
            dtype=torch.long,
        )

    def _update_path_stats(self, layer_name: str, expert_path: torch.Tensor, s: dict):
        """
        expert_path: CPU LongTensor of shape [batch, seq_len].
        """
        num_experts = s["transition_counts"].shape[0]
        batch_size, seq_len = expert_path.shape

        if seq_len == 0:
            return

        if (
            layer_name not in self.last_expert
            or self.last_expert[layer_name].numel() != batch_size
        ):
            self._reset_path_state_for_layer(layer_name, batch_size)

        last = self.last_expert[layer_name]
        active_run_len = self.active_run_len[layer_name]

        # Unique experts used within each observed path segment.
        for b in range(batch_size):
            s["unique_experts_sum"] += int(torch.unique(expert_path[b]).numel())
            s["num_paths_observed"] += 1

        for t in range(seq_len):
            cur = expert_path[:, t]

            valid_prev = last >= 0
            if valid_prev.any():
                prev_valid = last[valid_prev]
                cur_valid = cur[valid_prev]

                flat_trans = prev_valid * num_experts + cur_valid
                trans_counts = torch.bincount(
                    flat_trans,
                    minlength=num_experts * num_experts,
                ).reshape(num_experts, num_experts)

                s["transition_counts"] += trans_counts
                s["transition_count"] += int(valid_prev.sum().item())
                s["switch_count"] += int((prev_valid != cur_valid).sum().item())

            # Run-length tracking.
            # New run if no previous expert or expert changed.
            start_new_run = (~valid_prev) | (cur != last)
            s["run_count"] += int(start_new_run.sum().item())

            active_run_len[start_new_run] = 0
            active_run_len += 1

            current_max = int(active_run_len.max().item())
            s["max_run_len"] = max(s["max_run_len"], current_max)

            last.copy_(cur)

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

            with torch.no_grad():
                router_logits = router_logits.detach()
                probs = F.softmax(router_logits.float(), dim=-1)

                num_experts = moe_block.num_experts

                # You said k=1, so this is the actual routed expert.
                selected_prob, selected_expert = probs.max(dim=-1)

                expert_path = selected_expert.reshape(batch_size, seq_len).cpu()

                if layer_name not in self.stats:
                    self.stats[layer_name] = self._new_layer_stats(num_experts)

                s = self.stats[layer_name]

                top1_counts = torch.bincount(
                    selected_expert.cpu(),
                    minlength=num_experts,
                )

                weight_sums = torch.zeros(
                    num_experts,
                    dtype=torch.float32,
                    device=router_logits.device,
                )
                weight_sums.scatter_add_(
                    0,
                    selected_expert,
                    selected_prob.float(),
                )

                entropy = -(probs * probs.clamp_min(1e-20).log()).sum(dim=-1)
                max_prob = probs.max(dim=-1).values

                num_tokens = int(selected_expert.numel())

                s["tokens"] += num_tokens
                s["assignments"] += num_tokens
                s["num_calls"] += 1

                # For k=1, top-k counts are identical to top-1 counts.
                s["top1_counts"] += top1_counts
                s["topk_counts"] += top1_counts
                s["weight_sums"] += weight_sums.cpu()

                s["entropy_sum"] += float(entropy.sum().cpu())
                s["max_prob_sum"] += float(max_prob.sum().cpu())

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
            transition_count = max(s["transition_count"], 1)
            run_count = max(s["run_count"], 1)
            num_paths = max(s["num_paths_observed"], 1)

            top1_frac = s["top1_counts"].float() / tokens
            topk_frac = s["topk_counts"].float() / assignments
            avg_selected_weight = s["weight_sums"] / s["top1_counts"].clamp_min(1).float()

            transition_frac = s["transition_counts"].float() / transition_count
            switch_rate = s["switch_count"] / transition_count
            stay_rate = 1.0 - switch_rate
            mean_run_len = s["tokens"] / run_count
            avg_unique_experts_per_path = s["unique_experts_sum"] / num_paths

            out[layer_name] = {
                "num_calls": s["num_calls"],
                "tokens": s["tokens"],
                "assignments": s["assignments"],

                "top1_counts": s["top1_counts"].tolist(),
                "top1_frac": top1_frac.tolist(),
                "topk_counts": s["topk_counts"].tolist(),
                "topk_frac": topk_frac.tolist(),
                "avg_selected_weight": avg_selected_weight.tolist(),

                "avg_entropy": s["entropy_sum"] / tokens,
                "avg_max_prob": s["max_prob_sum"] / tokens,

                "transition_counts": s["transition_counts"].tolist(),
                "transition_frac": transition_frac.tolist(),
                "transition_count": s["transition_count"],
                "switch_count": s["switch_count"],
                "switch_rate": switch_rate,
                "stay_rate": stay_rate,
                "mean_run_len": mean_run_len,
                "max_run_len": s["max_run_len"],
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

            top1_counts = torch.zeros(num_experts, dtype=torch.long)
            topk_counts = torch.zeros(num_experts, dtype=torch.long)
            weight_sums = torch.zeros(num_experts, dtype=torch.float32)
            transition_counts = torch.zeros(num_experts, num_experts, dtype=torch.long)

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

                "transition_counts": transition_counts.tolist(),
                "transition_frac": (transition_counts.float() / transition_safe).tolist(),
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