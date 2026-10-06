import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_default_yaml_utf8_and_physics_flags():
    with open(ROOT / "config" / "default.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    assert cfg["data"]["source"] == "synthetic"
    assert cfg["model"]["use_sst"] is True
    assert cfg["model"]["use_sqg"] is True
    assert cfg["model"]["use_adv"] is True
    assert cfg["model"]["use_uncert"] is True
    assert "f0" in cfg["physics"]
    assert cfg["train"]["device"] in ("auto", "cpu", "cuda")


def test_paths_yaml_utf8():
    with open(ROOT / "config" / "paths.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    assert "natl60" in cfg
    assert "ssh_ref" in cfg["natl60"]
