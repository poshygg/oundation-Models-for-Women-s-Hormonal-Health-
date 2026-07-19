"""Smoke tests — 이 파일이 통과하면 환경 셋업이 최소한 굴러간다."""

from __future__ import annotations

import importlib

import pytest


CORE_MODULES = [
    "numpy",
    "pandas",
    "yaml",
    "dotenv",
    "hydra",
    "wandb",
    "mlflow",
]

PYTORCH_MODULES = [
    "torch",
    "torchvision",
    "transformers",
    "accelerate",
    "datasets",
    "timm",
]

TENSORFLOW_MODULES = [
    "tensorflow",
]


@pytest.mark.parametrize("name", CORE_MODULES)
def test_core_imports(name: str) -> None:
    importlib.import_module(name)


@pytest.mark.parametrize("name", PYTORCH_MODULES)
def test_pytorch_imports(name: str) -> None:
    importlib.import_module(name)


@pytest.mark.parametrize("name", TENSORFLOW_MODULES)
def test_tensorflow_imports(name: str) -> None:
    importlib.import_module(name)


def test_torch_cuda_reports() -> None:
    """GPU가 없어도 통과. cuda.is_available()가 실행만 되면 OK."""
    import torch

    _ = torch.cuda.is_available()  # 값은 True/False 무관 — 호출만 검증


def test_tf_device_query() -> None:
    import tensorflow as tf

    _ = tf.config.list_physical_devices()
