import json
import os
import random
from argparse import Namespace, ArgumentParser
from pathlib import Path
from pprint import pformat

import numpy as np
import torch

from utils import TypeJSON
from utils.logging_utils import get_logger

logger = get_logger(__name__)


def update_args(args: Namespace) -> None:
    config_path = Path(args.json_config_path)
    assert config_path.is_file(), logger.exception(f"config file {args.json_config_path} not found")
    logger.info("getting parameters from config.json")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
        args.__dict__.update(config)
        logger.info(pformat(vars(args)))


def set_seed(seed: int = 1234) -> None:
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
