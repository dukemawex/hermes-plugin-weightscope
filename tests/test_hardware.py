"""Hardware tests: real CUDA / MPS execution of the torch backend.

Marked ``hardware`` and deselected by default (pytest.ini: -m "not hardware"). Run them on a GPU machine with:
    python -m pytest -m hardware
They also skip individually when torch or the device is missing.
"""

import pytest

from test_backends import HAS_CUDA, HAS_MPS, _assert_parity, mixed_pair  # noqa: F401  (fixture re-export)
from weightscope.analysis import diff

pytestmark = pytest.mark.hardware

@pytest.mark.skipif(not HAS_CUDA, reason="no CUDA GPU available")
def test_cuda_parity(mixed_pair):  # pragma: no cover - no GPU on the dev box
    _assert_parity(diff(*mixed_pair, backend="numpy", top_k=100),
                   diff(*mixed_pair, backend="torch", device="cuda", top_k=100), rel=1e-9)


@pytest.mark.skipif(not HAS_MPS, reason="no Apple MPS device available")
def test_mps_parity(mixed_pair):  # pragma: no cover
    r = diff(*mixed_pair, backend="torch", device="mps", top_k=100)
    assert r["execution"]["compute_dtype"] == "float32"
    _assert_parity(diff(*mixed_pair, backend="numpy", top_k=100), r, rel=1e-4)
