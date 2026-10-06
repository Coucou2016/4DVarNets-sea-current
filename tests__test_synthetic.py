import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.synthetic import SyntheticOSSEConfig, generate_synthetic_osse, load_synthetic_npz


def test_generate_and_load(tmp_path):
    out = tmp_path / "syn.npz"
    d = generate_synthetic_osse(SyntheticOSSEConfig(n_time=10, height=16, width=16), out)
    assert d["ssh"].shape == (10, 16, 16)
    d2 = load_synthetic_npz(out)
    assert np.allclose(d["ssh"], d2["ssh"])
