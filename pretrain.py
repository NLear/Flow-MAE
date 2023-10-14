#!/usr/bin/env python3
# coding=utf-8
"""This is a sample Python script. """

import os
import sys
from typing import Dict, Tuple, List

import math

import datasets
import numpy as np
import torch
import transformers
from datasets import DatasetDict, Dataset
import torchvision
from transformers import (
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    ViTFeatureExtractor,
)
from transformers.trainer_utils import get_last_checkpoint, set_seed, EvalPrediction
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

from dataset.dataset_functions import PAYLOAD, PAYLOAD_LENGTH
from dataset.session_dataset import SessionDataSet
from model.mae import ViTMAEConfig, ViTMAEForPreTraining, ViTMAEForImageClassification
from pretrain.arguments import ModelArguments, DataTrainingArguments, CustomTrainingArguments
from pretrain.functions import collate_fn
from utils.logging_utils import get_logger

logger = get_logger(__name__)

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.22.0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/image-pretraining/requirements.txt")


def logger_setup(log_level):
    # Setup logging
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()


def get_last_ckpt(training_args: TrainingArguments):
    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )
    return last_checkpoint


def generate_config(model_args):
    # Load pretrained model and feature extractor
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.config_name:
        config = ViTMAEConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = ViTMAEConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        config = ViTMAEConfig(
            image_size=model_args.image_size,
            patch_size=model_args.patch_size,
            num_channels=model_args.num_channels,
            num_attention_heads=model_args.num_attention_heads,
            hidden_dropout_prob=model_args.hidden_dropout_prob
        )
        logger.warning("You are instantiating a new config instance from scratch.")
        if model_args.config_overrides is not None:
            logger.info(f"Overriding config: {model_args.config_overrides}")
            config.update_from_string(model_args.config_overrides)
            logger.info(f"New config: {config}")

    # adapt config
    config.update(
        {
            "mask_ratio": model_args.mask_ratio,
            "norm_pix_loss": model_args.norm_pix_loss,
        }
    )

    return config


def create_model(model_args, config):
    # create model
    if model_args.model_name_or_path:
        logger.info(f"load PreTrained model from {model_args.model_name_or_path}")
        model = ViTMAEForPreTraining.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    else:
        logger.info("Training new model from scratch")
        model = ViTMAEForPreTraining(config)
    return model


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, CustomTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    log_level = training_args.get_process_log_level()
    logger_setup(log_level)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    set_seed(1234)
    last_checkpoint = get_last_ckpt(training_args)
    config = generate_config(model_args)
    model = create_model(model_args, config)

    image_size = model.vit.embeddings.patch_embeddings.image_size
    num_pixels = image_size[0] * image_size[1]
    pacth_size = model.vit.embeddings.patch_size
    pixels_per_patch = pacth_size[0] * pacth_size[1]
    num_patches = model.vit.embeddings.num_patches
    train_ds = SessionDataSet(data_args.train_dir, num_pixels, pixels_per_patch, num_patches)
    test_ds = SessionDataSet(data_args.validation_dir, num_pixels, pixels_per_patch, num_patches)

    # Compute absolute learning rate
    total_train_batch_size = (
            training_args.train_batch_size * training_args.gradient_accumulation_steps * training_args.world_size
    )
    if training_args.base_learning_rate is not None:
        training_args.learning_rate = training_args.base_learning_rate * total_train_batch_size / 256

    # Initialize our trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds if training_args.do_train else None,
        eval_dataset=test_ds if training_args.do_eval else None,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    main()
