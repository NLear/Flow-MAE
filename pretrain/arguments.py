from dataclasses import dataclass, field
from typing import Optional, Tuple

from transformers import MODEL_FOR_MASKED_IMAGE_MODELING_MAPPING, TrainingArguments

MODEL_CONFIG_CLASSES = list(MODEL_FOR_MASKED_IMAGE_MODELING_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

from transformers import BertForSequenceClassification, BertConfig, BertTokenizer

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "Name of a dataset from the datasets package"}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    image_column_name: Optional[str] = field(
        default="layers_layerData", metadata={"help": "The column name of the images in the files."}
    )
    feature_len_column_name: Optional[str] = field(
        default="feature_len", metadata={"help": "The column name of the images in the files."}
    )
    label_column_name: Optional[str] = field(
        default="label", metadata={"help": "The column name of the images in the files."}
    )
    dataset_dir: Optional[str] = field(
        default=None, metadata={"help": "A folder containing the dataset."}
    )
    train_dir: Optional[str] = field(
        default=None, metadata={"help": "A folder containing the training data."}
    )
    validation_dir: Optional[str] = field(
        default=None, metadata={"help": "A folder containing the validation data."})
    train_val_split: Optional[float] = field(
        default=0.10,
        metadata={"help": "Percent to split off of train for validation."})
    max_train_samples: Optional[int] = field(
        default=None, metadata={
            "help": ("For debugging purposes or quicker training, truncate the number of training examples to this "
                     "value if set.")}, )
    max_eval_samples: Optional[int] = field(
        default=None, metadata={
            "help": ("For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                     "value if set.")}, )
    return_entity_level_metrics: bool = field(
        default=False, metadata={
            "help": "Whether to return all the entity levels during evaluation or just the overall ones."}, )

    # seed: int = field(default=1234, metadata={"help": "A seed for reproducible training."})

    def __post_init__(self):
        data_files = dict()
        if self.train_dir is not None:
            data_files["train"] = self.train_dir
        if self.validation_dir is not None:
            data_files["val"] = self.validation_dir
        self.data_files = data_files if data_files else None


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/feature extractor we are going to pre-train.
    """

    model_name_or_path: str = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name_or_path"}
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    feature_extractor_name: str = field(default=None, metadata={"help": "Name or path of preprocessor config."})
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    num_channels: int = field(
        default=1, metadata={"help": ""}
    )
    image_size: Tuple[int, int] = field(
        default=(1, 1024), metadata={"help": ""}
    )
    patch_size: Tuple[int, int] = field(
        default=(1, 64), metadata={"help": ""}
    )
    num_attention_heads: int = field(
        default=None, metadata={"help": ""}
    )
    hidden_dropout_prob: float = field(
        default=None, metadata={"help": "The ratio of the number of hidden_dropout_prob."}
    )
    attention_probs_dropout_prob: float = field(
        default=None, metadata={"help": "The ratio of the number of attention_probs_dropout_prob."}
    )
    mask_ratio: float = field(
        default=None, metadata={"help": "The ratio of the number of masked tokens in the input sequence."}
    )
    norm_pix_loss: bool = field(
        default=True, metadata={"help": "Whether or not to train with normalized pixel values as target."}
    )
    num_labels: int = field(
        default=None, metadata={"help": ""}
    )


@dataclass
class CustomTrainingArguments(TrainingArguments):
    base_learning_rate: float = field(
        default=5e-3, metadata={"help": "Base learning rate: absolute_lr = base_lr * total_batch_size / 256."}
    )
    skip_memory_metrics: bool = field(
        default=True, metadata={"help": "Whether or not to skip adding of memory profiler reports to metrics."}
    )
