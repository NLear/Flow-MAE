#!/usr/bin/env python3
# coding=utf-8
"""This is a sample Python script. """

import os
import sys
from functools import partial

import datasets
import numpy as np
import pandas as pd
import torch.distributed as dist
import transformers
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, classification_report,
    confusion_matrix)
from transformers import (
    HfArgumentParser,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint, set_seed
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

from dataset.session_dataset import dataset_collate_function, SessionDataSet
from model.mae import ViTMAEConfig, ViTMAEForPreTraining, ViTMAEForImageClassification
from pretrain.arguments import ModelArguments, DataTrainingArguments, CustomTrainingArguments
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
            hidden_dropout_prob=model_args.hidden_dropout_prob,
            attention_probs_dropout_prob=model_args.attention_probs_dropout_prob,
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

    return config


def create_model(model_args, config):
    if model_args.model_name_or_path:
        pretrained_model = ViTMAEForPreTraining.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        model = ViTMAEForImageClassification(config)
        model.vit.load_state_dict(pretrained_model.vit.state_dict())
    else:
        logger.info("Training new model from scratch")
        model = ViTMAEForImageClassification(config)
    return model


# Define our compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
# predictions and label_ids field) and has to return a dictionary string to float.
def compute_metrics(p, print_entity=False):
    """Computes accuracy on a batch of predictions"""
    labels = p.label_ids
    preds = np.argmax(p.predictions, axis=1)

    # Check if the current process is the main process
    is_main_process = not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0

    if print_entity and is_main_process:
        # Calculate and print confusion matrix
        conf_matrix = confusion_matrix(labels, preds)
        print("Confusion Matrix:")
        print(conf_matrix)

        # Create classification report dictionary and add class-wise accuracy
        class_report_dict = classification_report(labels, preds, output_dict=True, zero_division=0)
        # Convert dictionary to DataFrame and format columns
        class_report_df = pd.DataFrame(class_report_dict).transpose()
        class_report_df['support'] = class_report_df['support'].astype(int)
        # Print classification report with class-wise accuracy
        print("Classification Report with Class-wise Accuracy:")
        print(class_report_df)

    return {
        "accuracy_score": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, average='weighted'),
        "recall": recall_score(labels, preds, average='weighted'),
        "f1": f1_score(labels, preds, average='weighted'),
    }


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

    set_seed(training_args.seed)

    dataset = {}
    dataset['train'] = SessionDataSet(data_args.train_dir, 1024, 64, 16, mode="finetune")
    dataset['test'] = SessionDataSet(data_args.validation_dir, 1024, 64, 16, mode="finetune")
    model_args.num_labels = dataset['train'].num_labels

    # Count the number of labels.
    # dataset = datasets.load_from_disk(data_args.dataset_dir)
    # model_args.num_labels = dataset['train'].features['label'].num_classes

    last_checkpoint = get_last_ckpt(training_args)

    config = generate_config(model_args)
    model = create_model(model_args, config)

    pacth_size = model.vit.embeddings.patch_size
    pixels_per_patch = pacth_size[0] * pacth_size[1]
    num_patches = model.vit.embeddings.num_patches

    # train_ds = None
    # test_ds = None
    # if training_args.do_train:
    #     dataset_dict = datasets.load_dataset(str(data_args.train_dir))
    #     train_ds = dataset_dict[list(dataset_dict.keys())[0]]
    # if training_args.do_eval:
    #     dataset_dict = datasets.load_dataset(str(data_args.validation_dir))
    #     test_ds = dataset_dict[list(dataset_dict.keys())[0]]

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
        train_dataset=dataset['train'] if training_args.do_train else None,
        eval_dataset=dataset['test'] if training_args.do_eval else None,
        # data_collator=partial(dataset_collate_function, data_args=data_args),
        compute_metrics=compute_metrics,
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
        trainer.compute_metrics = partial(compute_metrics, print_entity=True)
        eval_result = trainer.evaluate()
        trainer.log_metrics("eval", eval_result)
        trainer.save_metrics("eval", eval_result)


if __name__ == "__main__":
    main()
