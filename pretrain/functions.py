from collections import defaultdict
from typing import Dict, Iterable, List

import torch
from datasets import Value
from torchvision.transforms import Normalize

from dataset.dataset_functions import STREAM, PAYLOAD, SEQ_HEIGHT, SEQ_LENGTH_BYTES, SEQ_LENGTH_HEX, MAX_EXAMPLES


def tensor_pad(example):
    example_tensor = [
        torch.cat(
            (torch.as_tensor(payload, dtype=torch.float32),
             torch.zeros(SEQ_LENGTH_BYTES - len(payload), dtype=torch.float32)
             ),
            dim=0
        )
        if len(payload) < SEQ_LENGTH_BYTES else torch.as_tensor(payload[:SEQ_LENGTH_BYTES], dtype=torch.float32)
        for payload in example
    ]
    example_pad_len = torch.stack(example_tensor, dim=0)
    if SEQ_HEIGHT - example_pad_len.size(0) > 0:
        pad_height = torch.zeros(
            (SEQ_HEIGHT - example_pad_len.size(0), SEQ_LENGTH_BYTES),
            dtype=torch.float32
        )
        example_pad_len_height = torch.cat((example_pad_len, pad_height), dim=0)

        return torch.unsqueeze(example_pad_len_height, dim=0)
    return torch.unsqueeze(example_pad_len, dim=0)


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    attention_mask = torch.stack([example["attention_mask"] for example in examples])
    return {
        "pixel_values": pixel_values,
        "attention_mask": attention_mask
    }


def get_mean_std(train_data, dim: int = 1):
    """
    Compute mean and variance for training data
    :param train_data: 自定义类Dataset(或ImageFolder即可)
    :param dim: 维度
    :return: (mean, std)
    """
    # print('Compute mean and variance for training data.')
    mean = torch.zeros(dim)
    std = torch.zeros(dim)
    sample_num = 0
    for X in train_data["pixel_values"]:
        sample_num += 1
        for d in range(dim):
            mean[d] = (mean[d] * (sample_num - 1) + X[d, :, :].mean()) / sample_num
            std[d] = (std[d] * (sample_num - 1) + X[d, :, :].std()) / sample_num
        if sample_num > 1_00:
            break
    # mean.div_(len(train_data))
    # std.div_(len(train_data))
    print(list(mean.numpy()), list(std.numpy()))


def getStat(train_data, dim: int = 1):
    '''
    Compute mean and variance for training data
    :param train_data: 自定义类Dataset(或ImageFolder即可)
    :return: (mean, std)
    '''
    print('Compute mean and variance for training data.')
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=1, shuffle=False, num_workers=0,
        pin_memory=True)
    mean = torch.zeros(dim)
    std = torch.zeros(dim)
    for X in train_loader:
        for d in range(dim):
            mean[d] += X["pixel_values"][:, d, :, :].mean()
            std[d] += X["pixel_values"][:, d, :, :].std()
    mean.div_(len(train_data))
    std.div_(len(train_data))
    return list(mean.numpy()), list(std.numpy())


def preprocess_images(example):
    normalizer = Normalize(mean=0.25, std=0.25)
    example["pixel_values"] = [
        normalizer(
            tensor_pad(example[PAYLOAD]).div(255)
        )
    ]
    return example


def multiprocess_merge(
        examples: Dict[str, Iterable[Value]],
        columns_to_keep: List[str] = None,
) -> Dict[str, Iterable[Value]]:
    columns = examples.keys()
    columns_to_keep = set(columns_to_keep).intersection(columns)
    columns_to_remove = set(columns).difference(columns_to_keep)

    merged_column: Dict[str, list] = defaultdict(list)
    overflow_to_sample_mapping: Dict[str: int] = {}
    merged_examples: Dict[str: list] = defaultdict(list)

    ex_columns = [examples[col] for col in columns_to_keep]
    for stream, payload in zip(examples[STREAM], examples[PAYLOAD]):
        merged_column[str(stream)].append(payload[:SEQ_LENGTH_HEX])

    for stream, payloads in merged_column.items():
        max_payloads = min(len(payloads), MAX_EXAMPLES)
        for i in range(0, max_payloads, SEQ_HEIGHT):
            p_slice = payloads[i: i + SEQ_HEIGHT]
            p_slice_uint8 = []
            for payload in p_slice:
                payload_int = [
                    int(payload[i: i + 2], 16)
                    for i in range(0, len(payload), 2)
                ]
                if len(payload_int) < SEQ_LENGTH_BYTES:
                    payload_int += [0] * (SEQ_LENGTH_BYTES - len(payload_int))
                p_slice_uint8.append(payload_int)
            if SEQ_HEIGHT - len(p_slice_uint8) > 0:
                p_slice_uint8.extend([[0] * SEQ_LENGTH_BYTES] * (SEQ_HEIGHT - len(p_slice_uint8)))
            merged_examples[PAYLOAD].append(p_slice_uint8)
            merged_examples[STREAM].append(int(stream))

    # for key, values in examples.items():
    #     if key is STREAM:
    #         merged_examples[str(key)] = [
    #             values[j]
    #             for j in range(overflow_to_sample_mapping[str(i)])
    #             for i in values
    #         ]
    #     elif key is PAYLOAD:
    #         merged_examples[PAYLOAD] = merged_payloads[key]

    return merged_examples


def multiprocess_merge_no_pad(
        examples: Dict[str, Iterable[Value]],
        columns_to_keep: List[str] = None,
) -> Dict[str, Iterable[Value]]:
    columns = examples.keys()
    columns_to_keep = set(columns_to_keep).intersection(columns)
    columns_to_remove = set(columns).difference(columns_to_keep)

    merged_column: Dict[str, list] = defaultdict(list)
    overflow_to_sample_mapping: Dict[str: int] = {}
    merged_examples: Dict[str: list] = defaultdict(list)

    ex_columns = [examples[col] for col in columns_to_keep]
    for stream, payload in zip(examples[STREAM], examples[PAYLOAD]):
        merged_column[str(stream)].append(payload[:SEQ_LENGTH_HEX])

    for stream, payloads in merged_column.items():
        max_payloads = min(len(payloads), MAX_EXAMPLES)
        for i in range(0, max_payloads, SEQ_HEIGHT):
            p_slice = payloads[i: i + SEQ_HEIGHT]
            p_slice_uint8 = []
            for payload in p_slice:
                payload_int = [
                    int(payload[i: i + 2], 16)
                    for i in range(0, len(payload), 2)
                ]
                p_slice_uint8.append(
                    torch.tensor(payload_int, dtype=torch.uint8)
                )
            merged_examples[STREAM].append(int(stream))
            merged_examples[PAYLOAD].append(p_slice_uint8)

    # for key, values in examples.items():
    #     if key is STREAM:
    #         merged_examples[str(key)] = [
    #             values[j]
    #             for j in range(overflow_to_sample_mapping[str(i)])
    #             for i in values
    #         ]
    #     elif key is PAYLOAD:
    #         merged_examples[PAYLOAD] = merged_payloads[key]

    return merged_examples
