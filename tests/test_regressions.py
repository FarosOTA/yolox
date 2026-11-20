"""Observable contracts covered by the foundation/correctness remediation."""

import json
import pickle
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch
from pycocotools.coco import COCO
from torch import nn

from yolox.core import Trainer, launch
from yolox.data.datasets import COCODataset
from yolox.exp import get_exp
from yolox.layers.fast_coco_eval_api import COCOeval_opt
from yolox.utils import ModelEMA, adjust_status, replace_module, resolve_device


def test_device_selection_and_single_device_launch(monkeypatch):
    for cuda, mps, expected in ((True, True, "cuda"), (False, True, "mps"),
                                (False, False, "cpu")):
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 2 if cuda else 0)
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)
        assert resolve_device().type == expected
        assert resolve_device("cpu").type == "cpu"
    for device in ("cuda", "mps"):
        with pytest.raises(RuntimeError, match="not available"):
            resolve_device(device)
    called = []
    launch(called.append, 1, args=("executed",), device="cpu")
    assert called == ["executed"]
    for devices, machines, rank in ((0, 1, 0), (2, 1, 0), (1, 2, 0), (1, 1, 1)):
        with pytest.raises(ValueError, match="one device"):
            launch(called.append, devices, machines, rank, args=("invalid",), device="cpu")
    assert called == ["executed"]


@pytest.mark.skipif(sys.platform == "win32", reason="RAM caching uses Unix fork")
@pytest.mark.parametrize("device", ["auto", "cuda"])
def test_cuda_ram_cache_launch_does_not_initialize_parent(tmp_path, device):
    # Use real forks in a fresh interpreter, with simulated CUDA runtime state.
    # A runtime initialized in the parent must reject use from its children.
    script = """
import importlib
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch
import torch
from yolox.utils import resolve_device

launcher = importlib.import_module('yolox.core.launch')
owner_pid = None
def cuda_available():
    global owner_pid
    if owner_pid is not None and owner_pid != os.getpid():
        raise RuntimeError('Cannot re-initialize CUDA in forked subprocess')
    owner_pid = os.getpid()
    return True

def worker(rank, main_func, *args):
    assert torch.cuda.is_available()
    main_func(rank)

def main(rank):
    Path(sys.argv[1], str(rank)).write_text('CUDA initialized in worker')

with patch.object(torch.cuda, 'is_available', cuda_available), \
     patch.object(torch.cuda, 'device_count', return_value=2), \
     patch.object(launcher, 'sys', SimpleNamespace(platform='linux')), \
     patch.object(launcher, '_distributed_worker', worker):
    device = resolve_device(sys.argv[2]).type
    launcher.launch(main, 2, args=(None, SimpleNamespace(cache='ram')), device=device)
"""
    result = subprocess.run([sys.executable, "-c", script, str(tmp_path), device],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    for rank in (0, 1):
        assert (tmp_path / str(rank)).read_text() == "CUDA initialized in worker"


@pytest.mark.parametrize("device", ["cpu", pytest.param(
    "mps", marks=pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires MPS")
)])
def test_training_cli_evaluation_and_resume(tmp_path, device):
    """Real COCO loader, training, EMA, validation and resume on non-CUDA backends."""
    (tmp_path / "annotations").mkdir()
    (tmp_path / "train2017").mkdir()
    (tmp_path / "val2017").mkdir()
    images, annotations = [], []
    for i in range(4):
        image = np.full((48, 64, 3), 70 + i * 20, dtype=np.uint8)
        image[9:31, 13:38] = (230, 80, 10)
        for split in ("train2017", "val2017"):
            cv2.imwrite(str(tmp_path / split / f"{i}.png"), image)
        images.append({"id": i, "file_name": f"{i}.png", "height": 48, "width": 64})
        if i % 2:
            annotations.append({"id": i, "image_id": i, "category_id": 4,
                                "bbox": [13, 9, 25, 22], "area": 550, "iscrowd": 0})
    data = {"images": images, "annotations": annotations,
            "categories": [{"id": 4, "name": "object"}]}
    for split in ("train2017", "val2017"):
        (tmp_path / "annotations" / f"instances_{split}.json").write_text(json.dumps(data))
    common = ["-n", "yolox-nano", "--device", device, "-b", "2", "-expn", "smoke"]
    overrides = ["data_dir", str(tmp_path), "output_dir", str(tmp_path / "runs"),
                 "input_size", "(64,64)", "test_size", "(64,64)", "num_classes", "1",
                 "data_num_workers", "0", "warmup_epochs", "0", "no_aug_epochs", "0",
                 "eval_interval", "1", "print_interval", "1", "multiscale_range", "0",
                 "test_conf", "0.0"]
    train = [sys.executable, "-m", "yolox.tools.train", *common]
    checkpoint = tmp_path / "runs/smoke/latest_ckpt.pth"
    previous = None
    for epoch in (1, 2):
        command = train + (["--resume"] if epoch == 2 else [])
        result = subprocess.run(command + overrides + ["max_epoch", str(epoch)],
                                capture_output=True, text=True)
        assert result.returncode == 0, result.stdout + result.stderr
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        assert saved["start_epoch"] == epoch and saved["ema_updates"] == 2 * epoch
        assert saved["optimizer"]["state"] and saved["scaler"] == {}
        weights = saved["training_model"]
        assert all(torch.isfinite(value).all() for value in weights.values())
        if previous is not None:
            assert any(not torch.equal(weights[key], previous[key]) for key in weights)
        previous = weights
        assert "Average Precision" in (tmp_path / "runs/smoke/train_log.txt").read_text()
    result = subprocess.run(
        [sys.executable, "-m", "yolox.tools.eval", *common, "-c", str(checkpoint), *overrides],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Average Precision" in (tmp_path / "runs/smoke/val_log.txt").read_text()


def test_experiment_overrides_and_file_identity(tmp_path):
    exp = get_exp(exp_name="yolox-nano")
    exp.merge(["ema", "False", "enable_mixup", "0", "test_size", "(96,64)"])
    assert exp.ema is False and exp.enable_mixup is False
    assert exp.test_size == (96, 64)
    with pytest.raises(ValueError):
        exp.merge(["ema", "not-a-boolean"])
    for number in (1, 2):
        directory = tmp_path / str(number)
        directory.mkdir()
        path = directory / "experiment.py"
        path.write_text("from yolox.exp import Exp as Base\n"
                        f"class Exp(Base):\n    identity = {number}\n")
        loaded = get_exp(exp_file=str(path))
        loaded.merge(["ema", "False"])
        serialized = pickle.dumps(loaded)
        restored = pickle.loads(serialized)
        assert restored.identity == number and restored.ema is False
        subprocess.run(
            [sys.executable, "-c",
             "import pickle, sys; exp = pickle.loads(sys.stdin.buffer.read()); "
             f"assert exp.identity == {number} and exp.ema is False"],
            input=serialized, check=True,
        )


def test_model_helpers_restore_and_replace_nested_modules():
    model = nn.Sequential(nn.Dropout(), nn.Sequential(nn.ReLU()))
    model[1].eval()
    with pytest.raises(RuntimeError), adjust_status(model, training=False):
        raise RuntimeError("failed inference")
    assert model.training and model[0].training and not model[1].training
    model = replace_module(model, nn.ReLU, nn.LeakyReLU,
                           lambda old, new: new(negative_slope=0.37))
    assert model[1][0].negative_slope == 0.37


def test_disk_cache_and_clipped_annotations(tmp_path):
    (tmp_path / "annotations").mkdir()
    (tmp_path / "train2017").mkdir()
    image = np.arange(12 * 20 * 3, dtype=np.uint8).reshape(12, 20, 3)
    cv2.imwrite(str(tmp_path / "train2017" / "a.png"), image)
    annotation = {
        "images": [{"id": 1, "file_name": "a.png", "height": 12, "width": 20}],
        "categories": [{"id": 4, "name": "object"}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 4,
                         "bbox": [-3, -2, 10, 7], "area": 70, "iscrowd": 0}],
    }
    (tmp_path / "annotations" / "instances_train2017.json").write_text(json.dumps(annotation))
    dataset = COCODataset(str(tmp_path), img_size=(12, 20), cache=True, cache_type="disk")
    actual, labels, _, _ = dataset.pull_item(0)
    np.testing.assert_array_equal(actual, image)
    np.testing.assert_array_equal(labels, [[0, 0, 7, 5, 0]])


@pytest.mark.parametrize("fallback", [False, True])
def test_coco_native_and_fallback_agree(fallback, monkeypatch):
    if fallback:
        def unavailable(*args, **kwargs):
            raise RuntimeError("compiler unavailable")
        monkeypatch.setattr("yolox.layers.fast_coco_eval_api.FastCOCOEvalOp.load", unavailable)
    gt = COCO()
    gt.dataset = {
        "images": [{"id": 1}], "categories": [{"id": 1}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 1,
                         "bbox": [1, 2, 10, 20], "area": 200, "iscrowd": 0}],
    }
    gt.createIndex()
    dt = gt.loadRes([{"image_id": 1, "category_id": 1,
                      "bbox": [1, 2, 10, 20], "score": 0.9}])
    evaluator = COCOeval_opt(gt, dt, "bbox")
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    assert evaluator.stats[0] == pytest.approx(1.0)


def test_checkpoint_resume_preserves_live_optimizer_and_ema(tmp_path):
    trainer = Trainer.__new__(Trainer)
    trainer.rank = 0
    trainer.device = "cpu"
    trainer.args = SimpleNamespace(logger="none", resume=True, ckpt=None, start_epoch=None)
    trainer.model = nn.Linear(2, 1)
    trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.1, momentum=0.9)
    trainer.scaler = torch.amp.GradScaler("cpu")
    trainer.use_model_ema = True
    trainer.ema_model = ModelEMA(trainer.model)
    trainer.ema_model.updates = 19
    trainer.file_name = str(tmp_path)
    trainer.epoch = 3
    trainer.best_ap = 0.6
    trainer.model(torch.tensor([[1., 3.]])).sum().backward()
    trainer.optimizer.step()
    live = {key: value.clone() for key, value in trainer.model.state_dict().items()}
    trainer.save_ckpt("latest")
    saved = torch.load(tmp_path / "latest_ckpt.pth", weights_only=True)
    assert not torch.equal(saved["model"]["weight"], saved["training_model"]["weight"])
    trainer.model = nn.Linear(2, 1)
    trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.9, momentum=0.1)
    trainer.scaler = torch.amp.GradScaler("cpu", init_scale=16)
    trainer.resume_train(trainer.model)
    for key, value in live.items():
        torch.testing.assert_close(trainer.model.state_dict()[key], value)
    assert trainer.optimizer.param_groups[0]["lr"] == 0.1
    momentum = next(iter(trainer.optimizer.state.values()))["momentum_buffer"]
    torch.testing.assert_close(momentum, torch.tensor([[1., 3.]]))
    assert trainer.resume_checkpoint["ema_updates"] == 19
    assert trainer.start_epoch == 4 and trainer.best_ap == 0.6
    assert trainer.scaler.get_scale() == 65536


def test_latest_keeps_updated_best_score(tmp_path):
    trainer = Trainer.__new__(Trainer)
    trainer.rank = 0
    trainer.args = SimpleNamespace(logger="none")
    trainer.model = nn.Linear(1, 1)
    trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.1)
    trainer.scaler = torch.amp.GradScaler("cpu", enabled=False)
    trainer.use_model_ema = False
    trainer.save_history_ckpt = False
    trainer.is_distributed = False
    trainer.file_name = str(tmp_path)
    trainer.epoch = 0
    trainer.best_ap = 0.4
    trainer.evaluator = None
    trainer.exp = SimpleNamespace(eval_interval=1, eval=lambda *a, **k: ((0.6, 0.7, ""), {}))
    trainer.after_epoch()
    assert torch.load(tmp_path / "latest_ckpt.pth", weights_only=True)["best_ap"] == 0.6
    trainer.exp.eval = lambda *a, **k: ((0.5, 0.6, ""), {})
    trainer.epoch = 1
    trainer.after_epoch()
    assert torch.load(tmp_path / "best_ckpt.pth", weights_only=True)["curr_ap"] == 0.6


@pytest.mark.skipif(sys.platform == "win32", reason="bundled OpenBLAS validation uses Unix executables")
def test_blas_validation_rejects_false_success(tmp_path):
    """Exercise the CLI with zero-exit failures, crashes and incomplete output."""
    source, build = tmp_path / "source", tmp_path / "build"
    fake = f"""#!{sys.executable}
import os, signal, sys
from pathlib import Path
name = Path(sys.argv[0]).name
mode = {str(tmp_path / 'mode')!r}
mode = Path(mode).read_text()
if name == 'openblas_utest':
    print('RESULTS: 2 tests (2 ok, 0 failed, 0 skipped)')
elif name.endswith('1'):
    print('----- PASS -----\\n' * 10)
else:
    cblas = name.startswith('x')
    text = ('SGEMV PASSED THE COLUMN-MAJOR COMPUTATIONAL TESTS ( 12 CALLS)\\n'
            'SGEMV PASSED THE ROW-MAJOR COMPUTATIONAL TESTS ( 12 CALLS)\\n'
            if cblas else 'SGEMV PASSED THE COMPUTATIONAL TESTS ( 12 CALLS)\\n')
    if name == 'xccblat2':
        text = 'COLUMN-MAJOR AND ROW-MAJOR DATA LAYOUTS ARE TESTED\\n'
        text += 'cblas_cgemv PASSED THE COMPUTATIONAL TESTS ( 12 CALLS)\\n' * (
            1 if mode == 'one-layout' else 2)
    text += 'END OF TESTS\\n'
    if name == 'sblat2':
        if mode == 'crash':
            os.kill(os.getpid(), signal.SIGTERM)
        if mode == 'fatal':
            text += 'FATAL ERROR - TESTS ABANDONED\\n'
        if mode == 'truncated':
            text = 'SGEMV PASSED THE TESTS OF ERROR-EXITS\\nEND OF TESTS\\n'
        if mode == 'missing':
            sys.exit(0)
    if cblas:
        print(text)
    else:
        Path(name.upper() + '.SUMM').write_text(text)
"""
    for directory in ("test", "ctest", "utest"):
        (source / directory).mkdir(parents=True)
        (build / directory).mkdir(parents=True)
    for dtype in "sdcz":
        for level in (1, 2, 3):
            for directory, name, data in (
                ("test", f"{dtype}blat{level}", f"{dtype}blat{level}.dat"),
                ("ctest", f"x{dtype}cblat{level}", f"{dtype}in{level}"),
            ):
                routine = "cblas_cgemv" if name == "xccblat2" else "SGEMV"
                (source / directory / data).write_text(f"{routine} T PUT F FOR NO TEST\n")
                executable = build / directory / name
                executable.write_text(fake)
                executable.chmod(0o755)
    executable = build / "utest/openblas_utest"
    executable.write_text(fake)
    executable.chmod(0o755)
    validator = Path(__file__).resolve().parents[1] / "demo/MegEngine/cpp/validate_openblas.py"
    for mode in ("valid", "fatal", "crash", "missing", "truncated", "one-layout"):
        (tmp_path / "mode").write_text(mode)
        result = subprocess.run(
            [sys.executable, str(validator), "--source", str(source), "--build", str(build),
             "--logs", str(tmp_path / mode)], capture_output=True, text=True,
        )
        assert result.returncode == (0 if mode == "valid" else 1), result.stdout
        assert ("25/25" if mode == "valid" else "24/25") in result.stdout


def test_megengine_weight_bridge_without_megengine(tmp_path):
    converter = Path(__file__).resolve().parents[1] / "demo/MegEngine/python/convert_weights.py"
    weights = {"conv.weight": torch.arange(6, dtype=torch.float32).reshape(2, 3),
               "bn.bias": torch.tensor([-1., 3.]), "bn.num_batches_tracked": torch.tensor(7)}
    checkpoint = tmp_path / "weights.pth"
    torch.save({"model": weights}, checkpoint)
    archive, copied = tmp_path / "weights.npz", tmp_path / "copied.npz"
    for source, destination in ((checkpoint, archive), (archive, copied)):
        subprocess.run([sys.executable, str(converter), "-w", str(source),
                        "-o", str(destination)], check=True)
    with np.load(copied, allow_pickle=False) as arrays:
        assert set(arrays.files) == set(weights)
        for name, expected in weights.items():
            np.testing.assert_array_equal(arrays[name], expected.numpy())
