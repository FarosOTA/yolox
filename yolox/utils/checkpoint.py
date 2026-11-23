#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.
import hashlib
import json
import os
from pathlib import Path
import shutil
from loguru import logger

import torch

from yolox.deploy.coco_classes import COCO_CLASSES


# Exact upstream release bytes, not filenames or an 80-class shape heuristic.
# URLs and verification procedure are recorded in docs/model_assets.md.
OFFICIAL_CHECKPOINTS = {
    "cd28f55fbbc1829f99d9ac9b38a16d259a22889739c8728ea877610201feff7b": "nano",
    "9de513de589ac98bb92d3bca53b5af7b9acfa9b0bacb831f7999d0f7afaee8f0": "tiny",
    "f55ded7181e1b0c13285c56e7790b8f0e8f8db590fe4edb37f0b7f345c913a30": "s",
    "60076992b32da82951c90cfa7bd6ab70eba9eda243e08b940a396f60ac2d19b6": "m",
    "1e6b7fa6240375370b2a8a8eab9066b3cdd43fd1d0bfa8d2027fd3a51def2917": "l",
    "5652330b6ae860043f091b8f550a60c10e1129f416edfdb65c259be6caf355cf": "x",
}


def validate_class_names(names):
    if (not isinstance(names, (list, tuple)) or not names
            or any(not isinstance(name, str) or not name.strip() for name in names)
            or len(set(names)) != len(names)):
        raise ValueError("Class names must be a non-empty list of unique non-empty strings")


def validate_class_initialization(names, mapping):
    validate_class_names(names)
    if not isinstance(mapping, dict):
        raise ValueError("classes.init_from_coco must be a target-to-COCO-name table")
    for target, source in mapping.items():
        sources = [source] if isinstance(source, str) else source
        if (target not in names or not isinstance(sources, list) or not sources
                or any(not isinstance(name, str) or name not in COCO_CLASSES for name in sources)
                or len(set(sources)) != len(sources)):
            raise ValueError(f"Invalid classes.init_from_coco mapping: {target!r} = {source!r}; "
                             "use a configured target and an exact COCO_CLASSES name or a "
                             "non-empty array of unique COCO_CLASSES names")


def checkpoint_class_names(path, class_file=None):
    with open(path, "rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    official = digest in OFFICIAL_CHECKPOINTS
    if official:
        names = list(COCO_CLASSES)
        if class_file is not None and json.loads(Path(class_file).read_text()) != names:
            raise ValueError("checkpoint_classes contradicts the verified official COCO order")
        logger.info("Verified official YOLOX-{} checkpoint: SHA-256 {}",
                    OFFICIAL_CHECKPOINTS[digest], digest)
    elif class_file is not None:
        names = json.loads(Path(class_file).read_text())
        validate_class_names(names)
        logger.warning("Checkpoint is not a registered official asset; trusting class order from {} "
                       "for SHA-256 {}. This file must belong to that checkpoint.", class_file, digest)
    else:
        raise ValueError("Unknown checkpoint taxonomy (SHA-256 " + digest + "). Only registered "
                         "official files imply COCO order. Set training.checkpoint_classes "
                         "(CLI: --checkpoint-classes) to the original run's classes.json; "
                         "filename and an 80-class head do not establish class meanings.")
    return names, official


def load_finetune_checkpoint(model, path, *, class_names=None, init_from_coco=None,
                            checkpoint_classes=None):
    """Load inference/EMA weights; remap named classes without changing checkpoint files."""
    if class_names is None:
        if checkpoint_classes is not None:
            raise ValueError("--checkpoint-classes requires --prepared-data with target class names")
        state = torch.load(path, map_location="cpu", weights_only=True)["model"]
        return load_ckpt(model, state)
    mapping = init_from_coco or {}
    validate_class_initialization(class_names, mapping)
    mapping = {name: [source] if isinstance(source, str) else source
               for name, source in mapping.items()}
    source_names, official = checkpoint_class_names(path, checkpoint_classes)
    state = torch.load(path, map_location="cpu", weights_only=True)["model"]
    if state and all(key.startswith("module.") for key in state):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    target = model.state_dict()
    classifiers = {f"head.cls_preds.{level}.{kind}"
                   for level in range(3) for kind in ("weight", "bias")}
    if state.keys() != target.keys() or not classifiers.issubset(state):
        raise ValueError("Checkpoint architecture mismatch: model tensor keys differ")
    for key, value in state.items():
        expected = target[key]
        if key in classifiers:
            if expected.shape[0] != len(class_names):
                raise ValueError("Target model class count does not match classes.names")
            if value.shape != (len(source_names), *expected.shape[1:]):
                raise ValueError(f"Checkpoint classifier/class-list mismatch at {key}: "
                                 f"got {tuple(value.shape)}, expected "
                                 f"{(len(source_names), *expected.shape[1:])}")
        elif value.shape != expected.shape:
            raise ValueError(f"Checkpoint architecture mismatch at {key}: "
                             f"{tuple(value.shape)} != {tuple(expected.shape)}")
    rows = []
    for index, name in enumerate(class_names):
        sources = mapping.get(name, []) if official else ([name] if name in source_names else [])
        if not official and name not in source_names and name in mapping:
            raise ValueError(f"Custom checkpoint has no class {name!r}; its COCO initializer "
                             "cannot be applied from this custom file. Start from an official "
                             "checkpoint or remove that mapping to initialize the new row randomly.")
        source_indices = [source_names.index(source) for source in sources]
        rows.append(source_indices)
        logger.info("Class {} ({}) <- {}", index, name,
                    f"{'COCO' if official else 'custom'} {sources!r} {source_indices}"
                    if source_indices else "fresh random weights / prior bias")
        if len(source_indices) > 1:
            logger.warning("Class {!r}: averaging COCO weight/bias rows is a warm start, "
                           "not an exact union of source predictions", name)
    # Retain model-constructor initialization for fresh rows; never reset objectness here.
    loaded = {key: value for key, value in state.items() if key not in classifiers}
    for key in classifiers:
        value = target[key].clone()
        for index, source_indices in enumerate(rows):
            if source_indices:
                source = (state[key][source_indices[0]] if len(source_indices) == 1
                          else state[key][source_indices].mean(dim=0))
                value[index].copy_(source)
        loaded[key] = value
    model.load_state_dict(loaded, strict=True)
    return model


def load_ckpt(model, ckpt):
    model_state_dict = model.state_dict()
    if ckpt and not model_state_dict.keys() & ckpt.keys() and all(
        key.startswith("module.") for key in ckpt
    ):
        ckpt = {key.removeprefix("module."): value for key, value in ckpt.items()}
    load_dict = {}
    for key_model, v in model_state_dict.items():
        if key_model not in ckpt:
            logger.warning(
                "{} is not in the ckpt. Please double check and see if this is desired.".format(
                    key_model
                )
            )
            continue
        v_ckpt = ckpt[key_model]
        if v.shape != v_ckpt.shape:
            logger.warning(
                "Shape of {} in checkpoint is {}, while shape of {} in model is {}.".format(
                    key_model, v_ckpt.shape, key_model, v.shape
                )
            )
            continue
        load_dict[key_model] = v_ckpt

    model.load_state_dict(load_dict, strict=False)
    return model


def save_checkpoint(state, is_best, save_dir, model_name=""):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    filename = os.path.join(save_dir, model_name + "_ckpt.pth")
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(save_dir, "best_ckpt.pth")
        shutil.copyfile(filename, best_filename)
