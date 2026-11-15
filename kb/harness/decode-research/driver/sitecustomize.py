import json
import os
import threading

"""Decode-research T1 driver (harness-side; production code untouched).

Activated ONLY when FREETOKEN_DECODE_RESEARCH=<outdir> is set (the runner passes
it plus PYTHONPATH to the ft serve tree, so plain boots are unaffected).

Decode steps run as CUDA-graph REPLAYS: model host code (incl.
OffloadMoELayer._decode_routed) executes only during capture, not per step.
Step hook = GraphRunner.replay; the _decode_routed hook only serves the eager
run (--cuda-graph-max-bs 0).

Per-step stream spans: at the END of replay call k we record _e0 (enqueued
right after step k's ops); at the END of call k+1 we record _e1 and store the
pair; spans are computed once after a synchronize in _close_window.

ARMOR: every hook body is wrapped in try/except -- a driver bug must degrade
to a no-op, never kill the engine (the KeyError incident, 2026-09-26).
"""

_OUT = os.environ.get("FREETOKEN_DECODE_RESEARCH")
if _OUT:
    try:
        import torch
        from freetoken.engine.engine import Engine
        from freetoken.engine.graph import GraphRunner
        from freetoken.layers import moe as _moe_mod

        _TAG = os.environ.get("FREETOKEN_DR_TAG", "run")
        _SSTART = int(os.environ.get("FREETOKEN_DR_SSTART", "150"))
        _SEND = int(os.environ.get("FREETOKEN_DR_SEND", "160"))
        _LSTART = _SSTART * 42
        _LEND = _SEND * 42
        _ENGINES = []
        _st = {
            "steps": 0, "layers": 0, "prof": None, "win": False,
            "dumped": False, "dead": False, "lock": threading.Lock(),
            "step_pairs": [], "pending_e0": None,
        }

        # --- per-step routing trace (FREETOKEN_DR_ROUTES=1; eager decode only) ---
        # topk_ids is rewritten IN PLACE by ensure_experts_hybrid, so the copy lands
        # BEFORE the original call: the buffer holds raw expert ids, never slot ids.
        # n_replay_steps in the dump lets the consumer assert the boot was eager.
        _ROUTES = os.environ.get("FREETOKEN_DR_ROUTES", "") == "1"
        _RT_MAX_CALLS = 200000
        _ROUTES_DUMP_EVERY = 2688  # 64 decode steps x 42 MoE layers: rewrite cadence only
        _rt = {"buf": None, "n": 0, "ids": [], "dead": False}

        def _cache_meta():
            out = []
            for eng in _ENGINES:
                for c in _engine_caches(eng):
                    meta = {
                        "num_layers": getattr(c, "num_layers", None),
                        "num_experts": getattr(c, "num_experts", None),
                        "cache_size": getattr(c, "cache_size", None),
                        "decode_target": getattr(c, "decode_target", None),
                        "pool_sizes": list(getattr(c, "pool_sizes", None) or []),
                        "pool_of_layer": list(getattr(c, "pool_of_layer", None) or []),
                        "hybrid_max_fetch": getattr(c, "hybrid_max_fetch", None),
                        "hybrid_fetch_fraction": list(getattr(c, "hybrid_fetch_fraction", None) or []) or None,
                        "max_slots": getattr(c, "max_slots", None),
                        "locked_hybrid": sorted(getattr(c, "locked_hybrid", None) or []),
                        "cpu_layer_ids": sorted(getattr(c, "cpu_layer_ids", None) or []),
                        "unpinned_layers": sorted(getattr(c, "_unpinned_layers", None) or []),
                    }
                    try:
                        meta["layer_layout"] = list(c.layer_layout)
                        meta["slot_bytes_per_layer"] = (
                            [int(c._copy_feat_bytes[g]) for g in c.layer_layout]
                            if c._copy_feat_bytes is not None else None
                        )
                    except Exception as e:
                        meta["layout_err"] = repr(e)
                    # authoritative per-layer slot bytes live on the Engine
                    # (layer_slot_bytes(banks.sources)); _copy_feat_bytes stays None
                    # when the fused-copy plan is not built
                    engine_bytes = getattr(eng, "_moe_layer_slot_bytes", None)
                    if engine_bytes and not meta.get("slot_bytes_per_layer"):
                        try:
                            meta["slot_bytes_per_layer"] = [int(b) for b in list(engine_bytes)]
                        except Exception as e:
                            meta["engine_bytes_err"] = repr(e)
                    out.append(meta)
            return out

        def _dump_routes():
            # callers hold _st["lock"]; no re-acquire here
            if _rt["buf"] is None or _rt["n"] == 0:
                return
            try:
                torch.cuda.synchronize()
                doc = {
                    "tag": _TAG,
                    "n_calls": _rt["n"],
                    "layer_ids": _rt["ids"],
                    "routes": _rt["buf"][: _rt["n"]].cpu().tolist(),
                    "n_replay_steps": _st["steps"],
                    "meta": _cache_meta(),
                }
                path = os.path.join(_OUT, "routes_%s.json" % _TAG)
                with open(path, "w") as f:
                    json.dump(doc, f)
                print("[decode-research] DUMPED routes -> %s (calls=%d)" % (path, _rt["n"]), flush=True)
            except Exception as e:
                print("[decode-research] routes dump FAILED: %r" % (e,), flush=True)

        if _ROUTES:
            import atexit

            atexit.register(_dump_routes)
            # LIFO: the final stats rewrite runs before the final routes dump;
            # lambda because _dump is defined further below (name resolves at call time)
            atexit.register(lambda: _dump())

        def _engine_caches(eng):
            # Engine keeps the singular moe_offload_cache; the plural list lives on
            # GraphRunner (same fallback as kvcache/cache_status._moe_offload_caches)
            caches = getattr(eng, "moe_offload_caches", None)
            if caches:
                return list(caches)
            single = getattr(eng, "moe_offload_cache", None)
            return [single] if single is not None else []

        def _dump(trace_path=None):
            if not _ENGINES:
                # helper processes (tokenizer etc.) arm too but never build an Engine;
                # their exit dumps must not clobber the engine process's stats file
                return
            caches = []
            for eng in _ENGINES:
                for c in _engine_caches(eng):
                    entry = {
                        "num_layers": c.num_layers,
                        "decode_target": getattr(c, "decode_target", None),
                        "cache_size": getattr(c, "cache_size", None),
                    }
                    for key, fn in (
                        ("aggregate", c.decode_miss_stats),
                        ("per_layer", c.decode_miss_stats_per_layer),
                        ("routing", c.decode_routing_stats),
                    ):
                        try:
                            entry[key] = fn()
                        except Exception as e:
                            entry[key + "_err"] = repr(e)
                    caches.append(entry)
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            spans = []
            for pair in _st["step_pairs"]:
                try:
                    spans.append(round(pair[0].elapsed_time(pair[1]), 3))
                except Exception:
                    pass
            doc = {
                "tag": _TAG,
                "n_replay_steps": _st["steps"],
                "n_layer_calls": _st["layers"],
                "step_spans_ms": spans,
                "caches": caches,
            }
            path = os.path.join(_OUT, "stats_%s.json" % _TAG)
            with open(path, "w") as f:
                json.dump(doc, f, indent=1)
            print("[decode-research] DUMPED stats -> %s (trace=%s)" % (path, trace_path), flush=True)

        def _close_window():
            trace_path = None
            try:
                prof = _st["prof"]
                if prof is not None:
                    prof.stop()
                    trace_path = os.path.join(_OUT, "trace_%s.json" % _TAG)
                    prof.export_chrome_trace(trace_path)
            except Exception as e:
                print("[decode-research] profiler stop/export failed: %r" % (e,), flush=True)
                trace_path = None
            try:
                _dump(trace_path)
            except Exception as e:
                print("[decode-research] dump FAILED: %r" % (e,), flush=True)
            _st["dumped"] = True
            _st["prof"] = None
            print("[decode-research] window CLOSED (steps=%d pairs=%d)"
                  % (_st["steps"], len(_st["step_pairs"])), flush=True)
            if _ROUTES:
                _dump_routes()

        def _patched_init(self, config):
            caches = _orig_init(self, config)
            try:
                # Engine keeps the cache singular (moe_offload_cache); the plural attr
                # never exists here, so the naive getattr silently flipped nothing
                for c in _engine_caches(self):
                    c.collect_stats = True
                    c.collect_decode_freq = True
                _ENGINES.append(self)
            except Exception as e:
                print("[decode-research] init-hook error (flags may be off): %r" % (e,), flush=True)
            return caches

        _orig_init = Engine._init_offload_moe_cache
        Engine._init_offload_moe_cache = _patched_init

        # --- replay hook (graphed decode steps) ---
        _orig_replay = GraphRunner.replay

        def _replay(self, batch):
            out = _orig_replay(self, batch)
            try:
                with _st["lock"]:
                    if _st["dumped"] or _st["dead"]:
                        return out
                    if _st["win"]:
                        e1 = torch.cuda.Event(enable_timing=True)
                        e1.record()
                        _st["step_pairs"].append((_st["pending_e0"], e1))
                    _st["steps"] += 1
                    n = _st["steps"]
                    if n == _SSTART and not _st["win"]:
                        _st["win"] = True
                        if os.environ.get("FREETOKEN_DR_PROF") == "1":
                            try:
                                from torch.profiler import ProfilerActivity, profile

                                p = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA])
                                p.start()
                                _st["prof"] = p
                                print("[decode-research] profiler window OPEN at step=%d" % n, flush=True)
                            except Exception as e:
                                print("[decode-research] profiler UNAVAILABLE (%r); events-only" % (e,), flush=True)
                                _st["prof"] = None
                        else:
                            print("[decode-research] events-only window OPEN at step=%d" % n, flush=True)
                    if _st["win"]:
                        _st["pending_e0"] = torch.cuda.Event(enable_timing=True)
                        _st["pending_e0"].record()
                    if n >= _SEND and _st["win"]:
                        _close_window()
            except Exception as e:
                # never let the driver kill the engine
                print("[decode-research] replay-hook error (driver disabled): %r" % (e,), flush=True)
                _st["dead"] = True
            return out

        GraphRunner.replay = _replay

        # --- eager-decode fallback hook (no graphs) ---
        _orig_routed = _moe_mod.OffloadMoELayer._decode_routed

        def _routed(self, hidden_states, topk_weights, topk_ids):
            if _ROUTES and not _rt["dead"]:
                try:
                    with _st["lock"]:
                        if _rt["buf"] is None:
                            _rt["buf"] = torch.zeros(
                                (_RT_MAX_CALLS, topk_ids.numel()), dtype=torch.int32, device=topk_ids.device
                            )
                            print("[decode-research] routes buffer armed (%d x %d)"
                                  % (_RT_MAX_CALLS, topk_ids.numel()), flush=True)
                        if _rt["n"] < _RT_MAX_CALLS:
                            _rt["buf"][_rt["n"]].copy_(topk_ids.detach().reshape(-1).to(torch.int32))
                            _rt["ids"].append(int(self.layer_id))
                            _rt["n"] += 1
                            if _rt["n"] % _ROUTES_DUMP_EVERY == 0:
                                _dump()
                                _dump_routes()
                except Exception as e:
                    print("[decode-research] routes capture error (routes disabled): %r" % (e,), flush=True)
                    _rt["dead"] = True
            out = _orig_routed(self, hidden_states, topk_weights, topk_ids)
            try:
                with _st["lock"]:
                    if _st["dumped"] or _st["dead"]:
                        return out
                    _st["layers"] += 1
                    n = _st["layers"]
                    if n == _LSTART and not _st["win"] and _st["steps"] < _SSTART:
                        _st["win"] = True
                        if os.environ.get("FREETOKEN_DR_PROF") == "1":
                            try:
                                from torch.profiler import ProfilerActivity, profile

                                p = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA])
                                p.start()
                                _st["prof"] = p
                                print("[decode-research] eager profiler window OPEN at layer-call=%d" % n, flush=True)
                            except Exception as e:
                                print("[decode-research] profiler UNAVAILABLE (%r); events-only" % (e,), flush=True)
                                _st["prof"] = None
                        else:
                            print("[decode-research] eager events-only window OPEN at layer-call=%d" % n, flush=True)
                    if n >= _LEND and _st["win"]:
                        _close_window()
            except Exception as e:
                print("[decode-research] routed-hook error (driver disabled): %r" % (e,), flush=True)
                _st["dead"] = True
            return out

        _moe_mod.OffloadMoELayer._decode_routed = _routed
        print("[decode-research] driver armed: steps %d..%d layer-calls %d..%d tag=%s out=%s"
              % (_SSTART, _SEND, _LSTART, _LEND, _TAG, _OUT), flush=True)
        if _ROUTES:
            print("[decode-research] routes mode ON (cadence %d calls + window close + exit)" % _ROUTES_DUMP_EVERY, flush=True)
    except Exception as e:
        # not an engine-capable process (tokenizer / helper workers): stay inert,
        # but say WHY - a silent inert arm cost a full boot (2026-11-15)
        print("[decode-research] driver INERT: %r" % (e,), flush=True)
