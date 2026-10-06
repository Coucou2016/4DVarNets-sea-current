import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.natl60 import (
    NATL60_URLS,
    NATL60_TIME_EPOCH,
    check_natl60_paths,
    indices_for_split,
    load_natl60,
    missing_files_message,
)


def test_import_does_not_require_files():
    assert "obs" in NATL60_URLS
    assert "ssh_ref" in NATL60_URLS


def test_check_missing_paths():
    missing = check_natl60_paths({"obs": str(ROOT / "does_not_exist.nc")})
    assert any("ssh_ref" in m for m in missing)
    assert any("obs" in m for m in missing)
    # refs present but obs/oi still required in paper mode
    assert check_natl60_paths(
        {
            "ssh_ref": str(ROOT / "missing_ssh.nc"),
            "sst_ref": str(ROOT / "missing_sst.nc"),
            "u_ref": str(ROOT / "missing_u.nc"),
            "v_ref": str(ROOT / "missing_v.nc"),
        }
    )
    # debug mode: refs only
    missing_dbg = check_natl60_paths(
        {
            "ssh_ref": str(ROOT / "missing_ssh.nc"),
            "sst_ref": str(ROOT / "missing_sst.nc"),
            "u_ref": str(ROOT / "missing_u.nc"),
            "v_ref": str(ROOT / "missing_v.nc"),
        },
        allow_truth_background=True,
    )
    assert all(not m.startswith("obs:") and not m.startswith("oi:") for m in missing_dbg)


def test_load_natl60_missing_raises_with_urls():
    paths = {k: str(ROOT / "missing" / f"{k}.nc") for k in NATL60_URLS}
    with pytest.raises(FileNotFoundError, match="download_natl60"):
        load_natl60(paths)
    msg = missing_files_message(paths)
    assert "https://" in msg


def test_paper_split_indices_no_val_train_overlap():
    times = np.arange("2012-10-01", "2013-10-01", dtype="datetime64[D]")
    train = set(indices_for_split(times, "train", dT=7))
    val = set(indices_for_split(times, "val", dT=7))
    test = set(indices_for_split(times, "test", dT=7))
    assert train
    assert val
    assert test
    assert train.isdisjoint(val)
    assert times[min(test)] >= np.datetime64("2012-10-20")
    assert times[max(train)] <= np.datetime64("2013-09-30")


def test_center_crop_and_max_samples_helpers():
    from data.dataset import _cap_indices, _center_crop_spatial

    data = {
        "ssh": np.zeros((4, 20, 20), dtype=np.float32),
        "u": np.zeros((4, 20, 20), dtype=np.float32),
        "v": np.zeros((4, 20, 20), dtype=np.float32),
        "sst": np.zeros((4, 20, 20), dtype=np.float32),
        "y_ssh": np.zeros((4, 20, 20), dtype=np.float32),
        "mask_ssh": np.ones((4, 20, 20), dtype=np.float32),
        "mask_sst": np.ones((4, 20, 20), dtype=np.float32),
        "time": np.arange(4),
    }
    cropped = _center_crop_spatial(data, 8)
    assert cropped["ssh"].shape == (4, 8, 8)
    assert len(_cap_indices(list(range(100)), 10)) == 10
    assert _cap_indices(list(range(5)), 10) == list(range(5))


def test_natl60_seconds_epoch_decoding():
    secs = np.array([43200.0, 129600.0])
    times = NATL60_TIME_EPOCH + (secs * 1e9).astype("timedelta64[ns]")
    assert times[0].astype("datetime64[D]") == np.datetime64("2012-10-01")
    assert times[1].astype("datetime64[D]") == np.datetime64("2012-10-02")
