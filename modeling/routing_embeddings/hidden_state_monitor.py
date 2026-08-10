from contextlib import contextmanager
import torch


class FfnHiddenStateMonitor:
    """
    Captures the hidden state feeding into the FFN (mlp) block at every
    layer of the model, once per query.

    For N queries and an L-layer model, `all_states` ends up as a list of N
    dicts, each mapping FFN layer name -> hidden state tensor, one entry
    per layer (so len(all_states[i]) == L).

    Uses the same hooking approach as MoeRoutingMonitor: walk
    model.named_modules(), find the FFN submodule at each layer (matched by
    a name suffix, default ".mlp"), and attach a hook there. The difference
    is we use register_forward_pre_hook instead of register_forward_hook,
    since we want that submodule's *input* hidden state, not its output -
    no tuple-unpacking needed, inputs[0] is directly the hidden state.

    A "query" can involve multiple internal forward() calls - e.g.
    model.generate() does one prefill pass over the full prompt, then one
    decode pass per generated token, and every one of those calls will
    fire the hook on every layer. `capture` controls which call's hidden
    state is kept per layer:
      - "first" (default): the prefill / first call, i.e. the hidden state
        produced while encoding the full prompt. This is almost always
        what "the hidden state for this query" means, and it's what gives
        you exactly one tensor per layer per query.
      - "last": the most recent call. In a generate() loop this ends up
        being the final decode step - a single new token's hidden state.
      - "all": keep every call's hidden state in a list per layer (the
        full per-forward-call trace, at correspondingly higher memory
        cost).

    `pool` optionally collapses the sequence dimension of each captured
    tensor:
      - "none" (default): keep the full [batch, seq_len, hidden] tensor.
      - "last_token": keep only the last token's hidden state, [batch, hidden].
      - "mean": mean-pool over the sequence dimension, [batch, hidden].
    """

    def __init__(
        self,
        model,
        layer_suffix: str = ".mlp",
        capture: str = "first",
        pool: str = "none",
        store_device: str = "cpu",
        store_dtype: torch.dtype = None,
    ):
        assert capture in ("first", "last", "all")
        assert pool in ("none", "last_token", "mean")

        self.model = model
        self.layer_suffix = layer_suffix
        self.capture = capture
        self.pool = pool
        self.store_device = store_device
        self.store_dtype = store_dtype

        self.handles = []
        self._recording = False
        self._current_query_states = {}

        self.all_states = []     # list[dict[layer_name -> Tensor | list[Tensor]]]
        self.query_labels = []   # index-aligned with all_states; whatever you pass to end_query()/query()

    # ---- capture plumbing ----------------------------------------------

    def _pool(self, hs: torch.Tensor) -> torch.Tensor:
        if self.pool == "none":
            return hs
        if hs.ndim == 3:      # [batch, seq, hidden]
            dim = 1
        elif hs.ndim == 2:    # [seq, hidden] (no batch dim)
            dim = 0
        else:
            return hs
        if self.pool == "last_token":
            return hs.select(dim, hs.shape[dim] - 1)
        if self.pool == "mean":
            return hs.mean(dim=dim)
        return hs

    def _store(self, hs: torch.Tensor) -> torch.Tensor:
        hs = hs.detach()
        if self.store_dtype is not None:
            hs = hs.to(self.store_dtype)
        # .clone() so we own this memory - without it we could be holding a
        # view into a buffer the model reuses or frees on a later layer /
        # the next forward call (especially with KV-cache / in-place ops).
        return hs.clone().to(self.store_device)

    def _make_hook(self, layer_name: str):
        def hook(module, inputs):
            if not self._recording:
                return
            if not inputs or not torch.is_tensor(inputs[0]):
                # If your model passes hidden_states as a kwarg rather than
                # positionally, switch to:
                #   module.register_forward_pre_hook(hook, with_kwargs=True)
                # and read kwargs["hidden_states"] instead of inputs[0].
                return

            hs = self._store(self._pool(inputs[0]))

            if self.capture == "first":
                self._current_query_states.setdefault(layer_name, hs)
            elif self.capture == "last":
                self._current_query_states[layer_name] = hs
            else:  # "all"
                self._current_query_states.setdefault(layer_name, []).append(hs)

        return hook

    # ---- registration ----------------------------------------------

    def register(self):
        self.remove()

        print("[FfnHiddenStateMonitor] FFN blocks found:")
        for name, module in self.model.named_modules():
            if name.endswith(self.layer_suffix):
                print("  ", name)
                handle = module.register_forward_pre_hook(self._make_hook(name))
                self.handles.append(handle)

        print(f"[FfnHiddenStateMonitor] registered {len(self.handles)} hooks")
        return self

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    # ---- per-query lifecycle ----------------------------------------------

    def start_query(self):
        self._current_query_states = {}
        self._recording = True

    def end_query(self, label=None):
        self._recording = False
        states = self._current_query_states
        self._current_query_states = {}
        self.all_states.append(states)
        self.query_labels.append(label)
        return states

    @contextmanager
    def query(self, label=None):
        """
        with mon.query("prompt about cats"):
            model.generate(**inputs)
        states = mon.all_states[-1]   # dict: layer_name -> Tensor
        """
        self.start_query()
        try:
            yield
        finally:
            self.end_query(label=label)

    # ---- convenience runner ----------------------------------------------

    def capture_queries(self, run_fn, queries):
        """
        run_fn: callable(query) -> None. Does the actual model(...) /
                model.generate(...) call for a single query. The monitor
                is already recording when run_fn is called, so you don't
                need to touch start_query/end_query yourself.
        queries: iterable of whatever run_fn expects (raw text, tokenized
                 inputs, etc).

        Returns self.all_states: list[dict[layer_name -> Tensor]], one
        dict per query, in the same order as `queries`.
        """
        for q in queries:
            with self.query(label=q if isinstance(q, str) else None):
                run_fn(q)
        return self.all_states

    def reset(self):
        self.all_states = []
        self.query_labels = []
        self._current_query_states = {}
        self._recording = False

    # ---- inspection ----------------------------------------------

    def describe(self, query_index: int = -1):
        states = self.all_states[query_index]
        label = self.query_labels[query_index]
        print(f"[FfnHiddenStateMonitor] query {query_index} ({label}): {len(states)} layers")
        for name, hs in states.items():
            if isinstance(hs, list):
                shapes = [tuple(t.shape) for t in hs]
                preview = shapes[:3] + (["..."] if len(shapes) > 3 else [])
                print(f"  {name}: {len(hs)} calls, shapes={preview}")
            else:
                print(f"  {name}: shape={tuple(hs.shape)}, dtype={hs.dtype}")