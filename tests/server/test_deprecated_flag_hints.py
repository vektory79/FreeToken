"""The deprecated --nvfp4-backend warns at parse time when the strategy decodes on the CPU.

The engine's KernelSelectionError stays the authoritative gate; the hint just moves the fix
before the model build. It belongs to the deprecated flag alone: an explicit --quant-backend
entry is a deliberate choice and gets no second-guessing.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

import freetoken.server.args as args_module
from freetoken.server.args import parse_args

ANON_PATH = "/models/anon"
HINT = "cannot run with --moe-strategy"


class _Config:
    def to_dict(self):
        return {"architectures": ["Qwen3MoeForCausalLM"], "torch_dtype": "bfloat16"}


class _Warnings:
    """Captures the module logger's warnings; freetoken loggers do not propagate to root, so caplog sees nothing."""

    def __init__(self):
        self.messages = []
        self.debugs = []

    def warning(self, msg, *args):
        self.messages.append(msg % args if args else msg)

    def debug(self, msg, *args):
        self.debugs.append(msg % args if args else msg)

    def __getattr__(self, name):
        return lambda *args, **kwargs: None


def _parse(monkeypatch, extra):
    warnings = _Warnings()
    monkeypatch.setattr(args_module, "logger", warnings)
    with patch("freetoken.utils.cached_load_hf_config", lambda _p: _Config()):
        args, _ = parse_args(["--model", ANON_PATH, *extra])
    return args, [m for m in warnings.messages if HINT in m]


@pytest.mark.parametrize("prefix", (
    ["--moe-strategy", "hybrid"],
    ["--moe-strategy", "cpu"],
    ["--moe-backend", "hybrid"],  # the user's exact failing invocation: both deprecated flags
    ["--moe-strategy", "offload", "--moe-cpu-layers", "8"],  # offload commits the CPU executor too, mirroring engine _decode_target
))
def test_deprecated_nvfp4_backend_hints_at_the_cpu_capable_kernel(monkeypatch, prefix):
    args, hints = _parse(monkeypatch, [*prefix, "--nvfp4-backend", "flashinfer"])
    assert args.quant_backend == "moe.nvfp4=b12x"
    assert any("'b12x' cannot run with --moe-strategy" in m and "moe.nvfp4=triton" in m for m in hints), hints


def test_explicit_quant_backend_gets_no_hint(monkeypatch):
    args, hints = _parse(monkeypatch, ["--moe-strategy", "hybrid", "--quant-backend", "moe.nvfp4=b12x"])
    assert args.quant_backend == "moe.nvfp4=b12x"
    assert not hints


def test_gpu_decode_strategy_gets_no_hint(monkeypatch):
    for extra in (["--moe-strategy", "offload"], []):  # the bare default resolves to auto, which decodes on the GPU
        args, hints = _parse(monkeypatch, [*extra, "--nvfp4-backend", "flashinfer"])
        assert args.quant_backend == "moe.nvfp4=b12x"
        assert not hints


def test_deprecated_nvfp4_backend_with_a_cpu_capable_kernel_gets_no_hint(monkeypatch):
    """A value that already decodes on the CPU is a deliberate pin, not a mistake to warn about."""
    args, hints = _parse(monkeypatch, ["--moe-strategy", "hybrid", "--nvfp4-backend", "triton"])
    assert args.quant_backend == "moe.nvfp4=triton"
    assert not hints


def test_mutually_exclusive_flags_exit_before_any_hint(monkeypatch):
    """--nvfp4-backend next to --quant-backend is a usage error; the usage error must win, with no hint before it."""
    warnings = _Warnings()
    monkeypatch.setattr(args_module, "logger", warnings)
    with patch("freetoken.utils.cached_load_hf_config", lambda _p: _Config()), pytest.raises(SystemExit):
        parse_args(["--model", ANON_PATH, "--moe-strategy", "hybrid", "--nvfp4-backend", "flashinfer", "--quant-backend", "moe.nvfp4=triton"])
    assert not [m for m in warnings.messages if HINT in m]


def test_registry_failure_skips_the_hint_silently(monkeypatch):
    """The capability answer comes from the kernel registry; when that probe cannot answer, parse must not."""
    from freetoken.layers.quantization import registry

    def boom(*_):
        raise NotImplementedError("no quant method for moe.nvfp4")

    monkeypatch.setattr(registry, "method_class", boom)
    warnings = _Warnings()
    monkeypatch.setattr(args_module, "logger", warnings)
    args_module._warn_gpu_only_nvfp4("moe.nvfp4=b12x", "hybrid", None)
    assert not warnings.messages
    assert warnings.debugs and "hint" in warnings.debugs[0]

    class _Half:
        candidates = (object(),)  # iterating it trips on cls.name -> AttributeError, same silent skip

    monkeypatch.setattr(registry, "method_class", lambda *_, **__: _Half)
    args_module._warn_gpu_only_nvfp4("moe.nvfp4=b12x", "hybrid", None)
    assert not warnings.messages