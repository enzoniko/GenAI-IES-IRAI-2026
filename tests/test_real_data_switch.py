"""P5.3: the dataset=mafaulda flag must resolve end-to-end against the
DatasetBundle contract (stub bundle — shapes only, no real data on disk)."""
import torch

from src.configs import DataCfg
from src.data import get_dataset
from src.data.contract import DatasetBundle


def test_stub_bundle_satisfies_contract():
    cfg = DataCfg(name="mafaulda_stub", T=256)
    b = get_dataset(cfg)
    assert isinstance(b, DatasetBundle)
    assert b.meta["has_ground_truth"] is False
    a = b.arrays("train")
    assert a["raw"].shape[1:] == (4, 256)
    assert a["jitter_phys"] is None            # ground-truth-dependent code must skip
    batch = next(iter(b.loader("train", batch_size=4)))
    assert batch.raw.shape == (4, 4, 256) and batch.omega.shape == (4,)
    fs = b.fewshot_fault_sets(2, seed=0)
    assert set(fs) == {1, 2, 3}
    assert fs[1]["raw_phys"].shape[0] <= 2


def test_registry_rejects_unknown():
    import pytest
    with pytest.raises(ValueError):
        get_dataset(DataCfg(name="nope"))
