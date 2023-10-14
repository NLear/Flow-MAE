#!/usr/bin/env python3
# coding=utf-8
"""This is a sample Python script. """

import os
import sys
from typing import Dict, Tuple, List

import math

# import datasets
import numpy as np
import torch
import transformers
# from datasets import DatasetDict, Dataset
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
    # datasets.utils.logging.set_verbosity(log_level)
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
            "num_labels": model_args.num_labels,
        }
    )

    # # create feature extractor
    # if model_args.feature_extractor_name:
    #     feature_extractor = ViTFeatureExtractor.from_pretrained(model_args.feature_extractor_name, **config_kwargs)
    # elif model_args.model_name_or_path:
    #     feature_extractor = ViTFeatureExtractor.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    # else:
    #     feature_extractor = ViTFeatureExtractor(do_resize=False, do_normalize=True)

    return config, None


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

    config, feature_extractor = generate_config(model_args)

    mode = "pretrain"
    # create model
    if model_args.model_name_or_path:
        logger.info(f"load PreTrained model from {model_args.model_name_or_path}")
        model = ViTMAEForImageClassification.from_pretrained(
            model_args.model_name_or_path,
            config=config
        )
        # model = ViTMAEForPreTraining.from_pretrained(
        #     model_args.model_name_or_path,
        #     from_tf=bool(".ckpt" in model_args.model_name_or_path),
        #     config=config,
        #     cache_dir=model_args.cache_dir,
        #     revision=model_args.model_revision,
        #     use_auth_token=True if model_args.use_auth_token else None,
        # )
        # state_dict = model.vit.state_dict()
        # model = ViTMAEForImageClassification(config)
        # model.vit.load_state_dict(state_dict)
        mode = "finetune"
    else:
        logger.info("Training new model from scratch")
        model = ViTMAEForPreTraining(config)

    # ds = Dataset.load_from_disk(data_args.dataset_name)
    # ds = ds.rename_column(PAYLOAD, "pixel_values")
    # if isinstance(ds, Dataset):
    #     ds = DatasetDict({"train": ds})
    # logger.info(f"load dataset dict {ds}")
    #
    # # If we don't have a validation split, split off a percentage of train as validation.
    # assert isinstance(ds, DatasetDict)
    # train_val_split = data_args.train_val_split
    # train_val_split = None if "validation" in ds.keys() else train_val_split
    # if isinstance(train_val_split, float) and train_val_split > 0.0:
    #     split = ds["train"].train_test_split(train_val_split)
    #     ds["train"] = split["train"]
    #     ds["validation"] = split["test"]
    # logger.info(f"split dataset {ds}")
    #
    # if training_args.do_train:
    #     column_names = ds["train"].column_names
    # else:
    #     column_names = ds["validation"].column_names
    #
    # if data_args.image_column_name is not None:
    #     image_column_name = data_args.image_column_name
    # elif "tcp.payload" in column_names:
    #     image_column_name = "tcp.payload"
    # elif "udp.payload" in column_names:
    #     image_column_name = "udp.payload"
    # else:
    #     image_column_name = column_names[-1]
    #
    # image_size = model.vit.embeddings.patch_embeddings.image_size
    # pixels_per_image = image_size[0] * image_size[1]
    # pacth_size = model.vit.embeddings.patch_size
    # pixels_per_patch = pacth_size[0] * pacth_size[1]
    # num_patches = model.vit.embeddings.num_patches
    #
    # def preprocess_images(examples):
    #     normalize = torchvision.transforms.Normalize(mean=0.5, std=0.5)
    #     all_pixels = []
    #     for pixel_values in examples["pixel_values"]:
    #         pixel_values = torch.as_tensor(pixel_values, dtype=torch.float).view(1, 1, -1)
    #         pixel_values = normalize(pixel_values / 255)
    #         pixel_values = torch.nn.functional.pad(pixel_values, (0, pixels_per_image - pixel_values.shape[-1], 0, 0))
    #         all_pixels.append(pixel_values)
    #     examples["pixel_values"] = all_pixels
    #
    #     examples["attention_mask"] = []
    #     for payload_len in examples[PAYLOAD_LENGTH]:
    #         effective_patches = math.ceil(payload_len / pixels_per_patch)
    #         attention_mask = torch.as_tensor(
    #             [1] * effective_patches + [0] * (num_patches - effective_patches)
    #         ).unsqueeze(dim=0)
    #         examples["attention_mask"].append(attention_mask)
    #
    #     return examples
    #
    # if training_args.do_train:
    #     if "train" not in ds:
    #         raise ValueError("--do_train requires a train dataset")
    #     if data_args.max_train_samples is not None:
    #         ds["train"] = ds["train"].shuffle(seed=training_args.seed).select(range(data_args.max_train_samples))
    #     # Set the training transforms
    #     ds["train"].set_transform(preprocess_images)
    #
    # if training_args.do_eval:
    #     if "validation" not in ds:
    #         raise ValueError("--do_eval requires a validation dataset")
    #     if data_args.max_eval_samples is not None:
    #         ds["validation"] = (
    #             ds["validation"].shuffle(seed=training_args.seed).select(range(data_args.max_eval_samples))
    #         )
    #     # Set the validation transforms
    #     ds["validation"].set_transform(preprocess_images)

    pacth_size = model.vit.embeddings.patch_size
    pixels_per_patch = pacth_size[0] * pacth_size[1]
    num_patches = model.vit.embeddings.num_patches
    # train_ds = SessionDataSet(data_args.train_dir, 1024, pixels_per_patch, num_patches, mode)
    test_ds = SessionDataSet(data_args.validation_dir, 1024, pixels_per_patch, num_patches, mode)

    # Compute absolute learning rate
    total_train_batch_size = (
            training_args.train_batch_size * training_args.gradient_accumulation_steps * training_args.world_size
    )
    if training_args.base_learning_rate is not None:
        training_args.learning_rate = training_args.base_learning_rate * total_train_batch_size / 256

    # # Load the accuracy metric from the datasets package
    # accuracy = evaluate.load("accuracy", average='weighted')
    #
    # # Define our compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # # predictions and label_ids field) and has to return a dictionary string to float.
    # def compute_metrics(p: EvalPrediction):
    #     """Computes accuracy on a batch of predictions"""
    #     preds = np.argmax(p.predictions, axis=1)
    #     refs = p.label_ids
    #     return accuracy.compute(predictions=preds, references=refs)

    def align_predictions(predictions: np.ndarray, label_ids: np.ndarray) -> Tuple[List[int], List[int]]:
        preds = np.argmax(predictions, axis=1)
        return list(preds), list(label_ids)

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

    def compute_metrics(p: EvalPrediction) -> Dict:
        # preds_list, out_label_list = align_predictions(p.predictions, p.label_ids)
        out_label_list = p.label_ids
        preds_list = np.argmax(p.predictions, axis=1)

        if data_args.return_entity_level_metrics:
            # Unpack nested dictionaries
            final_results = {"accuracy_score": accuracy_score(out_label_list, preds_list)}
            cls_report = classification_report(out_label_list, preds_list, digits=5, output_dict=True)
            for key, value in cls_report.items():
                if isinstance(value, dict):
                    for n, v in value.items():
                        final_results[f"{key}_{n}"] = v
                else:
                    final_results[key] = value
            return final_results
        else:
            return {
                "accuracy_score": accuracy_score(out_label_list, preds_list),
                "precision": precision_score(out_label_list, preds_list, average='weighted'),
                "recall": recall_score(out_label_list, preds_list, average='weighted'),
                "f1": f1_score(out_label_list, preds_list, average='weighted'),
            }

    # Initialize our trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        # train_dataset=train_ds if training_args.do_train else None,
        eval_dataset=test_ds if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        # tokenizer=feature_extractor,
        # data_collator=collate_fn,
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
