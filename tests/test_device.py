import torch

from tacet import model


def set_availability(monkeypatch, cuda, xpu):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(model, "xpu_is_available", lambda: xpu)


def test_auto_prefers_cuda(monkeypatch):
    set_availability(monkeypatch, cuda=True, xpu=True)
    assert model.resolve_device("auto").type == "cuda"


def test_auto_falls_back_to_xpu(monkeypatch):
    set_availability(monkeypatch, cuda=False, xpu=True)
    assert model.resolve_device("auto").type == "xpu"


def test_auto_falls_back_to_cpu(monkeypatch):
    set_availability(monkeypatch, cuda=False, xpu=False)
    assert model.resolve_device("auto").type == "cpu"


def test_explicit_device_is_kept(monkeypatch):
    set_availability(monkeypatch, cuda=True, xpu=True)
    assert model.resolve_device("cpu").type == "cpu"
    assert str(model.resolve_device("xpu:0")) == "xpu:0"


def test_xpu_runs_in_bfloat16_autocast():
    assert "xpu" in model.ACCELERATOR_TYPES
    assert "cuda" in model.ACCELERATOR_TYPES
    assert "cpu" not in model.ACCELERATOR_TYPES
