import contextlib
import copy
import json
import os
import re
import sys
import traceback
from concurrent.futures import as_completed
from pathlib import Path
from typing import Dict, Callable, Union, Optional, List, Tuple, Any

import datasets
import fsspec
import numpy as np
from datasets import DatasetDict, Dataset, NamedSplit, Split, Features, TaskTemplate, concatenate_datasets
from datasets.table import Table
from datasets.utils.doc_utils import is_documented_by
from datasets.utils.typing import PathLike

from utils.logging_utils import get_logger
from utils.multiprocess import BoundProcessPoolExecutor
from . import config

logger = get_logger(__name__)


class DatasetDicts(dict):

    def _check_values_type(self):
        for dataset_dict in self.values():
            if not isinstance(dataset_dict, DatasetDict):
                raise TypeError(
                    f"Values in `DatasetDict` should be of type `Dataset` but got type '{type(dataset_dict)}'")

    def __getitem__(self, k) -> DatasetDict:
        if isinstance(k, (str, NamedSplit)) or len(self) == 0:
            return super().__getitem__(k)
        else:
            available_suggested_splits = [
                split for split in (Split.TRAIN, Split.TEST, Split.VALIDATION) if split in self
            ]
            suggested_split = available_suggested_splits[0] if available_suggested_splits else list(self)[0]
            raise KeyError(
                f"Invalid key: {k}. Please first select a split. For example: "
                f"`my_dataset_dictionary['{suggested_split}'][{k}]`. "
                f"Available splits: {sorted(self)}"
            )

    @property
    def data(self) -> Dict[str, Dict[str, Table]]:
        self._check_values_type()
        return {k: dataset_dict.data for k, dataset_dict in self.items()}

    @property
    def num_columns(self) -> Dict[str, Dict[str, int]]:
        """Number of columns in each split of the dataset_dict.

        """
        self._check_values_type()
        return {k: dataset_dict.num_columns for k, dataset_dict in self.items()}

    @property
    def num_rows(self) -> Dict[str, Dict[str, int]]:
        """Number of rows in each split of the dataset_dict (same as :func:`datasets.Dataset.__len__`).

        """
        self._check_values_type()
        return {k: dataset_dict.num_rows for k, dataset_dict in self.items()}

    @property
    def column_names(self) -> Dict[str, Dict[str, List[str]]]:
        """Names of the columns in each split of the dataset.

        """
        self._check_values_type()
        return {k: dataset.column_names for k, dataset in self.items()}

    @property
    def shape(self) -> Dict[str, Dict[str, Tuple[int]]]:
        """Shape of each split of the dataset (number of columns, number of rows).

        """
        self._check_values_type()
        return {k: dataset.shape for k, dataset in self.items()}

    def flatten(self, max_depth=16) -> "DatasetDicts":
        """Flatten the Apache Arrow Table of each split (nested features are flatten).
        Each column with a struct type is flattened into one column per struct field.
        Other columns are left unchanged.
        """
        self._check_values_type()
        return DatasetDicts({k: dataset.flatten(max_depth=max_depth) for k, dataset in self.items()})

    def unique(self, column: str) -> Dict[str, Dict[str, List]]:
        """Return a list of the unique elements in a column for each split.
        """
        self._check_values_type()
        return {k: dataset.unique(column) for k, dataset in self.items()}

    def __repr__(self):
        res = "\n".join([f"{k}: {v}" for k, v in self.items()])
        res = re.sub(r"^", " " * 4, res, 0, re.M)
        return f"DatasetDicts({{\n{res}\n}})"

    def cast(self, features: Features) -> "DatasetDicts":
        """Cast the dataset to a new set of features.
        The transformation is applied to all the datasets of the dataset dictionary.

        You can also remove a column using :func:`Dataset.map` with `feature` but :func:`cast_`
        is in-place (doesn't copy the data to a new dataset) and is thus faster.
        :param features:

        :return:
        """
        self._check_values_type()
        return DatasetDicts({k: dataset.cast(features=features) for k, dataset in self.items()})

    def cast_column(self, column: str, feature) -> "DatasetDicts":
        """Cast column to feature for decoding.

        Args:
            column (:obj:`str`): Column name.
            feature (:class:`Feature`): Target feature.

        Returns:
            :class:`DatasetDicts`
        """
        self._check_values_type()
        return DatasetDicts({k: dataset.cast_column(column=column, feature=feature) for k, dataset in self.items()})

    def remove_columns(self, column_names: Union[str, List[str]]) -> "DatasetDicts":
        """Remove one or several column(s) from each split in the dataset
        and the features associated to the column(s).

        Args:
            column_names (:obj:`Union[str, List[str]]`): Name of the column(s) to remove.
        """
        self._check_values_type()
        return DatasetDicts({k: dataset.remove_columns(column_names=column_names) for k, dataset in self.items()})

    def rename_column(self, original_column_name: str, new_column_name: str) -> "DatasetDicts":
        """
        Rename a column in the dataset and move the features associated to the
        original column under the new column name.
        The transformation is applied to all the datasets of the dataset dictionary.

        Args:
            original_column_name (:obj:`str`): Name of the column to rename.
            new_column_name (:obj:`str`): New name for the column.
        """
        self._check_values_type()
        return DatasetDicts(
            {
                k: dataset.rename_column(original_column_name=original_column_name, new_column_name=new_column_name)
                for k, dataset in self.items()
            }
        )

    def rename_columns(self, column_mapping: Dict[str, str]) -> "DatasetDicts":
        """
        Rename several columns in the dataset, and move the features associated to the original columns under
        the new column names.
        The transformation is applied to all the datasets of the dataset dictionary.

        Args:
            column_mapping (:obj:`Dict[str, str]`): A mapping of columns to rename to their new names

        Returns:
            :class:`DatasetDict`: A copy of the dataset with renamed columns
        """
        self._check_values_type()
        return DatasetDicts({k: dataset.rename_columns(column_mapping=column_mapping) for k, dataset in self.items()})

    def class_encode_column(self, column: str, include_nulls: bool = False) -> "DatasetDicts":
        """Casts the given column as :obj:``datasets.features.ClassLabel`` and updates the tables.

        Args: column (`str`): The name of the column to cast include_nulls (`bool`, default `False`): Whether to
        include null values in the class labels. If True, the null values will be encoded as the `"None"` class label.
        """
        self._check_values_type()
        return DatasetDicts(
            {k: dataset.class_encode_column(column=column, include_nulls=include_nulls) for k, dataset in self.items()}
        )

    @contextlib.contextmanager
    def formatted_as(
            self,
            type: Optional[str] = None,
            columns: Optional[List] = None,
            output_all_columns: bool = False,
            **format_kwargs,
    ):
        """To be used in a `with` statement. Set ``__getitem__`` return format (type and columns)
        The transformation is applied to all the datasets of the dataset dictionary.

        Args:
            type (:obj:`str`, optional):
                output type selected in [None, 'numpy', 'torch', 'tensorflow', 'pandas',
                'arrow'] None means ``__getitem__`` returns python objects (default)
            columns (:obj:`List[str]`, optional):
                columns to format in the output None means ``__getitem__`` returns all columns (default)
            output_all_columns (:obj:`bool`, default to False):
                keep un-formatted columns as well in the output (as python objects)
            **format_kwargs (additional keyword arguments):
                keywords arguments passed to the convert function like
                `np.array`, `torch.tensor` or `tensorflow.ragged.constant`.
        """
        old_format_type = {
            k: {kk: dataset._format_type for kk, dataset in dataset_dict.items()} for k, dataset_dict in self.items()
        }
        old_format_kwargs = {
            k: {kk: dataset._format_kwargs for kk, dataset in dataset_dict.items()} for k, dataset_dict in self.items()
        }
        old_format_columns = {
            k: {kk: dataset._format_columns for kk, dataset in dataset_dict.items()} for k, dataset_dict in self.items()
        }
        old_output_all_columns = {
            k: {kk: dataset._output_all_columns for kk, dataset in dataset_dict.items()} for k, dataset_dict in
            self.items()
        }

        print(old_format_type)
        print(old_format_columns)

        try:
            self.set_format(type, columns, output_all_columns, **format_kwargs)
            yield
        finally:
            for k, dataset_dict in self.items():
                for kk, dataset in dataset_dict.items():
                    dataset.set_format(
                        old_format_type[k][kk], old_format_columns[k][kk], old_output_all_columns[k][kk],
                        **(old_format_kwargs[k][kk])
                    )

    def set_format(
            self,
            type: Optional[str] = None,
            columns: Optional[List] = None,
            output_all_columns: bool = False,
            **format_kwargs,
    ):
        """``__getitem__`` return format (type and columns)
        The format is set for every dataset in the dataset dictionary

        Args: type (:obj:`str`, optional): output type selected in [None, 'numpy', 'torch', 'tensorflow', 'pandas',
        'arrow'] None means ``__getitem__`` returns python objects (default) columns (:obj:`List[str]`, optional):
        columns to format in the output. None means ``__getitem__`` returns all columns (default). output_all_columns
        (:obj:`bool`, default to False): keep un-formatted columns as well in the output (as python objects)
        **format_kwargs (additional keyword arguments): keywords arguments passed to the convert function like
        `np.array`, `torch.tensor` or `tensorflow.ragged.constant`.

        """
        self._check_values_type()
        for dataset in self.values():
            dataset.set_format(type=type, columns=columns, output_all_columns=output_all_columns, **format_kwargs)

    def reset_format(self):
        """Reset ``__getitem__`` return format to python objects and all columns.
        The transformation is applied to all the datasets of the dataset dictionary.

        Same as ``self.set_format()``
        """
        self._check_values_type()
        for dataset in self.values():
            dataset.set_format()

    def set_transform(
            self,
            transform: Optional[Callable],
            columns: Optional[List] = None,
            output_all_columns: bool = False,
    ):
        """Set ``__getitem__`` return format using this transform. The transform is applied on-the-fly on batches
        when ``__getitem__`` is called. The transform is set for every dataset in the dataset dictionary As
        :func:`datasets.Dataset.set_format`, this can be reset using :func:`datasets.Dataset.reset_format`

        Args: transform (:obj:`Callable`, optional): user-defined formatting transform, replaces the format defined
        by :func:`datasets.Dataset.set_format` A formatting function is a callable that takes a batch (as a dict) as
        input and returns a batch. This function is applied right before returning the objects in ``__getitem__``.
        columns (:obj:`List[str]`, optional): columns to format in the output If specified, then the input batch of
        the transform only contains those columns. output_all_columns (:obj:`bool`, default to False): keep
        un-formatted columns as well in the output (as python objects) If set to True, then the other un-formatted
        columns are kept with the output of the transform.

        """
        self._check_values_type()
        for dataset in self.values():
            dataset.set_format("custom", columns=columns, output_all_columns=output_all_columns, transform=transform)

    def with_format(
            self,
            type: Optional[str] = None,
            columns: Optional[List] = None,
            output_all_columns: bool = False,
            **format_kwargs,
    ) -> "DatasetDicts":
        """Set ``__getitem__`` return format (type and columns). The data formatting is applied on-the-fly.
        The format ``type`` (for example "numpy") is used to format batches when using ``__getitem__``.
        The format is set for every dataset in the dataset dictionary

        It's also possible to use custom transforms for formatting using :func:`datasets.Dataset.with_transform`.

        Contrary to :func:`datasets.DatasetDict.set_format`, ``with_format`` returns a new DatasetDict object with
        new Dataset objects.

        Args: type (:obj:`str`, optional): Either output type selected in [None, 'numpy', 'torch', 'tensorflow',
        'pandas', 'arrow']. None means ``__getitem__`` returns python objects (default) columns (:obj:`List[str]`,
        optional): columns to format in the output None means ``__getitem__`` returns all columns (default)
        output_all_columns (:obj:`bool`, default to False): keep un-formatted columns as well in the output (as
        python objects) **format_kwargs (additional keyword arguments): keywords arguments passed to the convert
        function like `np.array`, `torch.tensor` or `tensorflow.ragged.constant`.
        """
        dataset = copy.deepcopy(self)
        dataset.set_format(type=type, columns=columns, output_all_columns=output_all_columns, **format_kwargs)
        return dataset

    def with_transform(
            self,
            transform: Optional[Callable],
            columns: Optional[List] = None,
            output_all_columns: bool = False,
    ) -> "DatasetDicts":
        """Set ``__getitem__`` return format using this transform. The transform is applied on-the-fly on batches
        when ``__getitem__`` is called. The transform is set for every dataset in the dataset dictionary

            As :func:`datasets.Dataset.set_format`, this can be reset using :func:`datasets.Dataset.reset_format`.

            Contrary to :func:`datasets.DatasetDict.set_transform`, ``with_transform`` returns a new DatasetDict
            object with new Dataset objects.

            Args:
                transform (:obj:`Callable`, optional): user-defined formatting transform, replaces the format defined
                    by :func:`datasets.Dataset.set_format`
                    A formatting function is a callable that takes a batch (as a dict) as input and returns a batch.
                    This function is applied right before returning the objects in ``__getitem__``.
                columns (:obj:`List[str]`, optional): columns to format in the output
                    If specified, then the input batch of the transform only contains those columns.
                output_all_columns (:obj:`bool`, default to False): keep un-formatted columns as well in the output
                    (as python objects)
                    If set to True, then the other un-formatted columns are kept with the output of the transform.
        """
        dataset = copy.deepcopy(self)
        dataset.set_transform(transform=transform, columns=columns, output_all_columns=output_all_columns)
        return dataset

    def map(
            self,
            function: Optional[Callable] = None,
            with_indices: bool = False,
            with_rank: bool = False,
            input_columns: Optional[Union[str, List[str]]] = None,
            batched: bool = False,
            batch_size: Optional[int] = 1000,
            drop_last_batch: bool = False,
            remove_columns: Optional[Union[str, List[str]]] = None,
            keep_in_memory: bool = False,
            load_from_cache_file: bool = True,
            cache_file_names: Optional[Dict[str, Optional[str]]] = None,
            writer_batch_size: Optional[int] = 1000,
            features: Optional[Features] = None,
            disable_nullable: bool = False,
            fn_kwargs: Optional[dict] = None,
            num_proc: Optional[int] = None,
            desc: Optional[str] = None,
    ) -> "DatasetDicts":
        """Apply a function to all the elements in the table (individually or in batches)
        and update the table (if function does update examples).
        The transformation is applied to all the datasets of the dataset dictionary.

        Args:
            function (`callable`): with one of the following signature:
                - `function(example: Dict[str, Any]) -> Dict[str, Any]` if `batched=False` and `with_indices=False`
                - `function(example: Dict[str, Any], indices: int) -> Dict[str, Any]` if `batched=False` and `
                    with_indices=True`
                - `function(batch: Dict[str, List]) -> Dict[str, List]` if `batched=True` and `with_indices=False`
                - `function(batch: Dict[str, List], indices: List[int]) -> Dict[str, List]` if `batched=True` and
                    `with_indices=True`

                For advanced usage, the function can also return a `pyarrow.Table`.
                Moreover if your function returns nothing (`None`), then `map` will run your function and return the
                dataset unchanged.

            with_indices (`bool`, defaults to `False`): Provide example indices to `function`. Note that in this case
            the signature of `function` should be `def function(example, idx): ...`.
            with_rank (:obj:`bool`, default `False`): Provide process rank to `function`. Note that in this case the
                signature of `function` should be `def function(example[, idx], rank): ...`.
            input_columns (`Optional[Union[str, List[str]]]`, defaults to `None`): The columns to be passed into
                `function` as positional arguments. If `None`, a dict mapping to all formatted columns is passed as
                one argument.
            batched (`bool`, defaults to `False`): Provide batch of examples to `function`
            batch_size (:obj:`int`, optional, defaults to `1000`): Number of examples per batch provided to
            `function`  if `batched=True`
                `batch_size <= 0` or `batch_size == None`: Provide the full dataset as a single batch to `function`
            drop_last_batch (:obj:`bool`, default `False`): Whether a last batch smaller than the batch_size should be
                dropped instead of being processed by the function.
            remove_columns (`Optional[Union[str, List[str]]]`, defaults to `None`): Remove a selection of columns
            while  doing the mapping.
                Columns will be removed before updating the examples with the output of `function`,
                i.e. if `function`  is adding
                columns with names in `remove_columns`, these columns will be kept.
            keep_in_memory (`bool`, defaults to `False`): Keep the dataset in memory instead of writing it to a
            cache file.
            load_from_cache_file (`bool`, defaults to `True`): If a cache file storing the current computation from
            `function`
                can be identified, use it instead of recomputing.
            cache_file_names (`Optional[Dict[str, str]]`, defaults to `None`): Provide the name of a path for the
            cache file. It is used to store the
                results of the computation instead of the automatically generated cache file name.
                You have to provide one :obj:`cache_file_name` per dataset in the dataset dictionary.
            writer_batch_size (:obj:`int`, default `1000`): Number of rows per write operation for the cache file
            writer.
                This value is a good trade-off between memory usage during the processing, and processing speed.
                Higher value makes the processing do fewer lookups, lower value consume less temporary memory while
                running `.map()`.
            features (`Optional[datasets.Features]`, defaults to `None`): Use a specific Features to store the cache
                file instead of the automatically generated one.
            disable_nullable (`bool`, defaults to `False`): Disallow null values in the table.
            fn_kwargs (:obj:`Dict`, optional, defaults to `None`): Keyword arguments to be passed to `function`
            num_proc (:obj:`int`, optional, defaults to `None`): Number of processes for multiprocessing. By default,
                it  doesn't use multiprocessing.
            desc (:obj:`str`, optional, defaults to `None`): Meaningful description to be displayed alongside with
                the  progress bar while mapping examples.

        """
        self._check_values_type()
        return DatasetDicts(
            {
                k: dataset.map(
                    function=function,
                    with_indices=with_indices,
                    with_rank=with_rank,
                    input_columns=input_columns,
                    batched=batched,
                    batch_size=batch_size,
                    drop_last_batch=drop_last_batch,
                    remove_columns=remove_columns,
                    keep_in_memory=keep_in_memory,
                    load_from_cache_file=load_from_cache_file,
                    cache_file_name=cache_file_names[k],
                    writer_batch_size=writer_batch_size,
                    features=features,
                    disable_nullable=disable_nullable,
                    fn_kwargs=fn_kwargs,
                    num_proc=num_proc,
                    desc=desc,
                )
                for k, dataset in self.items()
            }
        )

    def filter(
            self,
            function,
            with_indices=False,
            input_columns: Optional[Union[str, List[str]]] = None,
            batched: bool = False,
            batch_size: Optional[int] = 1000,
            keep_in_memory: bool = False,
            load_from_cache_file: bool = True,
            cache_file_names: Optional[Dict[str, Optional[str]]] = None,
            writer_batch_size: Optional[int] = 1000,
            fn_kwargs: Optional[dict] = None,
            num_proc: Optional[int] = None,
            desc: Optional[str] = None,
    ) -> "DatasetDicts":
        """Apply a filter function to all the elements in the table in batches
        and update the table so that the dataset only includes examples according to the filter function.
        The transformation is applied to all the datasets of the dataset dictionary.

        Args:
            function (`callable`): with one of the following signature:
                - ``function(example: Dict[str, Any]) -> bool`` if ``with_indices=False, batched=False``
                - ``function(example: Dict[str, Any], indices: int) -> bool`` if ``with_indices=True, batched=False``
                - ``function(example: Dict[str, List]) -> List[bool]`` if ``with_indices=False, batched=True``
                - ``function(example: Dict[str, List], indices: List[int]) -> List[bool]`` if ``with_indices=True,
                    batched=True``
            with_indices (`bool`, defaults to `False`): Provide example indices to `function`.
                Note that in this case the signature of `function` should be `def function(example, idx): ...`.
            input_columns (`Optional[Union[str, List[str]]]`, defaults to `None`): The columns to be passed into
                `function` as positional arguments. If `None`, a dict mapping to all formatted columns is passed as
                one argument.
            batched (`bool`, defaults to `False`): Provide batch of examples to `function`
            batch_size (:obj:`int`, optional, defaults to `1000`): Number of examples per batch provided to
                `function`  if `batched=True`
                `batch_size <= 0` or `batch_size == None`: Provide the full dataset as a single batch to `function`
            keep_in_memory (`bool`, defaults to `False`): Keep the dataset in memory instead of writing it to a
                cache file.
            load_from_cache_file (`bool`, defaults to `True`): If a cache file storing the current computation from
                `function` can be identified, use it instead of recomputing.
            cache_file_names (`Optional[Dict[str, str]]`, defaults to `None`): Provide the name of a path for the cache
                file. It is used to store the results of the computation instead of the automatically generated cache
                file name. You have to provide one :obj:`cache_file_name` per dataset in the dataset dictionary.
            writer_batch_size (:obj:`int`, default `1000`): Number of rows per write operation for the cache file writer
                This value is a good trade-off between memory usage during the processing, and processing speed.
                Higher value makes the processing do fewer lookups, lower value consume less temporary memory
                while running `.map()`.
            fn_kwargs (:obj:`Dict`, optional, defaults to `None`): Keyword arguments to be passed to `function`
            num_proc (:obj:`int`, optional, defaults to `None`): Number of processes for multiprocessing. By default
                it doesn't use multiprocessing.
            desc (:obj:`str`, optional, defaults to `None`): Meaningful description to be displayed alongside with the
            progress bar while filtering examples.
        """
        self._check_values_type()
        return DatasetDicts(
            {
                k: dataset.filter(
                    function=function,
                    with_indices=with_indices,
                    input_columns=input_columns,
                    batched=batched,
                    batch_size=batch_size,
                    keep_in_memory=keep_in_memory,
                    load_from_cache_file=load_from_cache_file,
                    cache_file_name=cache_file_names[k],
                    writer_batch_size=writer_batch_size,
                    fn_kwargs=fn_kwargs,
                    num_proc=num_proc,
                    desc=desc,
                )
                for k, dataset in self.items()
            }
        )

    def sort(
            self,
            column: str,
            reverse: bool = False,
            kind: str = None,
            null_placement: str = "last",
            keep_in_memory: bool = False,
            load_from_cache_file: bool = True,
            indices_cache_file_names: Optional[Dict[str, Optional[str]]] = None,
            writer_batch_size: Optional[int] = 1000,
    ) -> "DatasetDicts":
        """Create a new dataset sorted according to a column.
        The transformation is applied to all the datasets of the dataset dictionary.

        Currently sorting according to a column name uses pandas sorting algorithm under the hood.
        The column should thus be a pandas compatible type (in particular not a nested type).
        This also means that the column used for sorting is fully loaded in memory (which should be fine in most cases).

        Args:
            column (:obj:`str`): column name to sort by.
            reverse (:obj:`bool`, default `False`): If True, sort by descending order rather then ascending.
            kind (:obj:`str`, optional): Pandas algorithm for sorting selected in {‘quicksort’, ‘mergesort’,
            ‘heapsort’,  ‘stable’},
                The default is ‘quicksort’. Note that both ‘stable’ and ‘mergesort’ use timsort under the covers and,
                in general, the actual implementation will vary with data type. The ‘mergesort’ option is retained
                for backwards compatibility.
            null_placement (:obj:`str`, default `last`):
                Put `None` values at the beginning if ‘first‘; ‘last‘ puts `None` values at the end.

                *New in version 1.14.2*
            keep_in_memory (:obj:`bool`, default `False`): Keep the sorted indices in memory instead of  writing it
                to a cache file.
            load_from_cache_file (:obj:`bool`, default `True`): If a cache file storing the sorted indices
                can be identified, use it instead of recomputing.
            indices_cache_file_names (`Optional[Dict[str, str]]`, defaults to `None`): Provide the name of a path
                for the cache file. It is used to store the
                indices mapping instead of the automatically generated cache file name.
                You have to provide one :obj:`cache_file_name` per dataset in the dataset dictionary.
            writer_batch_size (:obj:`int`, default `1000`): Number of rows per write operation for the cache file
            writer. Higher value gives smaller cache files, lower value consume less temporary memory.
        """
        self._check_values_type()
        if indices_cache_file_names is None:
            indices_cache_file_names = {k: None for k in self}
        return DatasetDicts(
            {
                k: dataset.sort(
                    column=column,
                    reverse=reverse,
                    kind=kind,
                    null_placement=null_placement,
                    keep_in_memory=keep_in_memory,
                    load_from_cache_file=load_from_cache_file,
                    indices_cache_file_name=indices_cache_file_names[k],
                    writer_batch_size=writer_batch_size,
                )
                for k, dataset in self.items()
            }
        )

    def shuffle(
            self,
            seeds: Optional[Union[int, Dict[str, Optional[int]]]] = None,
            seed: Optional[int] = None,
            generators: Optional[Dict[str, np.random.Generator]] = None,
            keep_in_memory: bool = False,
            load_from_cache_file: bool = True,
            indices_cache_file_names: Optional[Dict[str, Optional[str]]] = None,
            writer_batch_size: Optional[int] = 1000,
    ) -> "DatasetDicts":
        """Create a new Dataset where the rows are shuffled.

        The transformation is applied to all the datasets of the dataset dictionary.

        Currently shuffling uses numpy random generators.
        You can either supply a NumPy BitGenerator to use, or a seed to initiate NumPy's default random generator (
        PCG64).

        Args:
            seeds (`Dict[str, int]` or `int`, optional): A seed to initialize the default BitGenerator if
                ``generator=None``. If None, then fresh, unpredictable entropy will be pulled from the OS.
                If an int or array_like[ints] is passed, then it will be passed to SeedSequence to derive the initial
                BitGenerator state.
                You can provide one :obj:`seed` per dataset in the dataset dictionary.
            seed (Optional `int`): A seed to initialize the default BitGenerator if ``generator=None``. Alias for seeds
            (a `ValueError` is raised if both are provided).
            generators (Optional `Dict[str, np.random.Generator]`): Numpy random Generator to use to compute the
            permutation of the dataset rows.
                If ``generator=None`` (default), uses np.random.default_rng (the default BitGenerator (PCG64) of NumPy).
                You have to provide one :obj:`generator` per dataset in the dataset dictionary.
            keep_in_memory (`bool`, defaults to `False`): Keep the dataset in memory instead of writing it to a
                cache file.
            load_from_cache_file (`bool`, defaults to `True`): If a cache file storing the current computation from
                `function` can be identified, use it instead of recomputing.
            indices_cache_file_names (`Dict[str, str]`, optional): Provide the name of a path for the cache file.
                It is used to store the indices mappings instead of the automatically generated cache file name.
                You have to provide one :obj:`cache_file_name` per dataset in the dataset dictionary.
            writer_batch_size (:obj:`int`, default `1000`): Number of rows per write operation for the cache file
                writer. This value is a good trade-off between memory usage during the processing, and processing speed.
                Higher value makes the processing do fewer lookups, lower value consume less temporary memory while
                running `.map()`.

        Example:

        ```py
        >>> from datasets import load_dataset
        >>> ds = load_dataset("rotten_tomatoes")
        >>> ds["train"]["label"][:10]
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

        # set a seed
        >>> shuffled_ds = ds.shuffle(seed=42)
        >>> shuffled_ds["train"]["label"][:10]
        [0, 1, 0, 1, 0, 0, 0, 0, 0, 0]
        ```
        """
        self._check_values_type()
        if seed is not None and seeds is not None:
            raise ValueError("Please specify seed or seeds, but not both")
        seeds = seed if seed is not None else seeds
        if seeds is None:
            seeds = {k: None for k in self}
        elif not isinstance(seeds, dict):
            seeds = {k: seeds for k in self}
        if generators is None:
            generators = {k: None for k in self}
        if indices_cache_file_names is None:
            indices_cache_file_names = {k: None for k in self}
        return DatasetDicts(
            {
                k: dataset.shuffle(
                    seed=seeds[k],
                    generator=generators[k],
                    keep_in_memory=keep_in_memory,
                    load_from_cache_file=load_from_cache_file,
                    indices_cache_file_name=indices_cache_file_names[k],
                    writer_batch_size=writer_batch_size,
                )
                for k, dataset in self.items()
            }
        )

    def save_to_disk(self, dest_root: str):
        """
        Saves a dataset dict to a filesystem using either :class:`~filesystems.S3FileSystem` or
        ``fsspec.spec.AbstractFileSystem``.

        For :class:`Image` and :class:`Audio` data:

        If your images and audio files are local files, then the resulting arrow file will store paths to these files.
        If you want to include the bytes or your images or audio files instead, you must `read()` those files first.
        This can be done by storing the "bytes" instead of the "path" of the images or audio files:

        Args:
            dest_root (``str``): Path (e.g. `dataset/train`) or remote URI
                (e.g. `s3://my-bucket/dataset/train`) of the dataset dict directory where the dataset dict will be
                saved to.
            src_root:
        """
        fs = fsspec.filesystem("file")

        if Path(dest_root).exists():
            logger.warning(f"dest_root {dest_root} is not empty, will overwrite")

        os.makedirs(dest_root, exist_ok=True)

        json.dump(
            {"splits": list(self)},
            fs.open(Path(
                dest_root, config.DATASET_DICTS_JSON_FILENAME
            ).as_posix(), "w", encoding="utf-8"),
        )
        for k, dataset_dict in self.items():
            dataset_dict.save_to_disk(Path(dest_root, k), fs)

    @staticmethod
    def load_from_disk(dataset_dicts_path: str, keep_in_memory: Optional[bool] = None) -> "DatasetDicts":
        """
        Load a dataset that was previously saved using :meth:`save_to_disk` from a filesystem using either
        :class:`~filesystems.S3FileSystem` or ``fsspec.spec.AbstractFileSystem``.

        Args:
            dataset_dicts_path (:obj:`str`): Path (e.g. ``"dataset/train"``) or remote URI (e.g.
                ``"s3//my-bucket/dataset/train"``) of the dataset dict directory where the dataset dict will be loaded
                from.
            keep_in_memory (:obj:`bool`, default ``None``): Whether to copy the dataset in-memory. If `None`, the
                dataset will not be copied in-memory unless explicitly enabled by setting
                `datasets.config.IN_MEMORY_MAX_SIZE` to nonzero. See more details in the
                :ref:`load_dataset_enhancing_performance` section.

        Returns:
            :class:`DatasetDict`
        """
        dataset_dicts = DatasetDicts()
        fs = fsspec.filesystem("file")

        dataset_dicts_json_path = Path(dataset_dicts_path, config.DATASET_DICTS_JSON_FILENAME).as_posix()
        dataset_dict_json_path = Path(dataset_dicts_path, datasets.config.DATASETDICT_JSON_FILENAME).as_posix()
        dataset_info_path = Path(dataset_dicts_path, datasets.config.DATASET_INFO_FILENAME).as_posix()

        if not fs.isfile(dataset_dicts_json_path):
            if fs.isfile(dataset_info_path) or fs.isfile(dataset_dict_json_path):
                raise FileNotFoundError(
                    f"No such file or directory: '{dataset_dicts_json_path}'. "
                    f"Expected to load a DatasetDicts object, but got a DatasetDict or Dataset. "
                    f"Please use datasets.load_from_disk instead."
                )
            else:
                raise FileNotFoundError(
                    f"missing {dataset_dicts_json_path} DatasetDicts path"
                )
        else:
            with open(os.path.join(dataset_dicts_json_path), "r") as file:
                json_config = json.load(file)
                splits = json_config[config.SPLITS_KEY]
                dataset_dicts_split_path = (
                    Path(dataset_dicts_path, key).as_posix()
                    for key in splits
                )
                for split, split_path in zip(splits, dataset_dicts_split_path):
                    dataset_dicts[split] = DatasetDict.load_from_disk(
                        split_path, fs,
                        keep_in_memory=keep_in_memory
                    )
            return dataset_dicts

    @staticmethod
    def from_csv(
            path_or_paths: Dict[str, Dict[str, str]],
            features: Optional[Features] = None,
            cache_dir: str = None,
            keep_in_memory: bool = False,
            **kwargs,
    ) -> "DatasetDicts":
        """Create DatasetDicts from CSV file(s).

        Args:
            path_or_paths (dict of path-like): Path(s) of the CSV file(s).
            features (:class:`Features`, optional): Dataset features.
            cache_dir (str, optional, default="~/.cache/huggingface/datasets"): Directory to cache data.
            keep_in_memory (bool, default=False): Whether to copy the data in-memory.
            **kwargs (additional keyword arguments): Keyword arguments to be passed to :meth:`pandas.read_csv`.

        Returns:
            :class:`DatasetDict`
        """
        return DatasetDicts(
            {
                name: DatasetDict.from_csv(paths, features=features, cache_dir=cache_dir,
                                           keep_in_memory=keep_in_memory, **kwargs)
                for name, paths in path_or_paths.items()
            }
        )

    @staticmethod
    def from_json(
            path_or_paths: Dict[str, Dict[str, PathLike]],
            features: Optional[Features] = None,
            cache_dir: str = None,
            keep_in_memory: bool = False,
            **kwargs,
    ) -> "DatasetDicts":
        """
        Create DatasetDicts from JSON Lines file(s).

         Args:
             path_or_paths (path-like or list of path-like): Path(s) of the JSON Lines file(s).
             features (:class:`Features`, optional): Dataset features.
             cache_dir (str, optional, default="~/.cache/huggingface/datasets"): Directory to cache data.
             keep_in_memory (bool, default=False): Whether to copy the data in-memory.
             **kwargs (additional keyword arguments): Keyword arguments to be passed to :class:`JsonConfig`.

         Returns:
             :class:`DatasetDict`
        """
        return DatasetDicts(
            {
                name: DatasetDict.from_json(paths, features=features, cache_dir=cache_dir,
                                            keep_in_memory=keep_in_memory, **kwargs)
                for name, paths in path_or_paths.items()
            }
        )

    @staticmethod
    def from_parquet(
            path_or_paths: Dict[str, PathLike],
            features: Optional[Features] = None,
            cache_dir: str = None,
            keep_in_memory: bool = False,
            columns: Optional[List[str]] = None,
            **kwargs,
    ) -> "DatasetDicts":
        """Create DatasetDict from Parquet file(s).

        Args:
            path_or_paths (dict of path-like): Path(s) of the CSV file(s).
            features (:class:`Features`, optional): Dataset features.
            cache_dir (str, optional, default="~/.cache/huggingface/datasets"): Directory to cache data.
            keep_in_memory (bool, default=False): Whether to copy the data in-memory.
            columns (:obj:`List[str]`, optional): If not None, only these columns will be read from the file.
                A column name may be a prefix of a nested field, e.g. 'a' will select
                'a.b', 'a.c', and 'a.d.e'.
            **kwargs (additional keyword arguments): Keyword arguments to be passed to :class:`ParquetConfig`.

        Returns:
            :class:`DatasetDict`
        """
        return DatasetDicts(
            {
                name: DatasetDict.from_parquet(paths, features=features, cache_dir=cache_dir,
                                               keep_in_memory=keep_in_memory, **kwargs)
                for name, paths in path_or_paths.items()
            }
        )

    @staticmethod
    def from_text(
            path_or_paths: Dict[str, PathLike],
            features: Optional[Features] = None,
            cache_dir: str = None,
            keep_in_memory: bool = False,
            **kwargs,
    ) -> "DatasetDicts":
        """Create DatasetDict from text file(s).

        Args:
            path_or_paths (dict of path-like): Path(s) of the text file(s).
            features (:class:`Features`, optional): Dataset features.
            cache_dir (str, optional, default="~/.cache/huggingface/datasets"): Directory to cache data.
            keep_in_memory (bool, default=False): Whether to copy the data in-memory.
            **kwargs (additional keyword arguments): Keyword arguments to be passed to :class:`TextConfig`.

        Returns:
            :class:`DatasetDict`
        """
        return DatasetDicts(
            {
                name: DatasetDict.from_text(paths, features=features, cache_dir=cache_dir,
                                            keep_in_memory=keep_in_memory, **kwargs)
                for name, paths in path_or_paths.items()
            }
        )

    @is_documented_by(Dataset.prepare_for_task)
    def prepare_for_task(self, task: Union[str, TaskTemplate], id: int = 0) -> "DatasetDicts":
        self._check_values_type()
        return DatasetDicts(
            {k: dataset.prepare_for_task(task=task, id=id) for k, dataset in self.items()}
        )

    @is_documented_by(Dataset.align_labels_with_mapping)
    def align_labels_with_mapping(self, label2id: Dict, label_column: str) -> "DatasetDicts":
        self._check_values_type()
        return DatasetDicts(
            {
                k: dataset.align_labels_with_mapping(label2id=label2id, label_column=label_column)
                for k, dataset in self.items()
            }
        )

    @staticmethod
    def _dict_batch_loader(input_dict: Dict[str, Any], batch_size: int) -> List[Tuple[str, Any]]:
        assert batch_size > 0
        cnt = 0
        batch = []
        for k, v in input_dict.items():
            batch.append((k, v))
            cnt += 1
            if cnt % batch_size == 0:
                yield batch
                batch = []
        if batch:
            yield batch

    @staticmethod
    def _process_executor(
            dataset_func: Callable[..., DatasetDict],
            input_dict: DatasetDict,
            function: Callable = None,
            num_proc: int = 16,
            batch_size: int = 16,
            **kwargs
    ) -> DatasetDict:
        """
        Read a Dataset or DatasetDict from source directory and then
        map Dataset to single process function call or
        map DatasetDict to num_proc parallel function call

        Args:
            src (str): root directory of Dataset or DatasetDict
            input_dict: DatasetDict to map
            function (Callable): the function to be applied to each dataset_dict

        Returns:
            :class:`Dataset` or :class:`DatasetDict`:
            - If `dataset_path` is a path of a dataset_dict directory: the dataset_dict requested.
            - If `dataset_path` is a path of a dataset_dict dict directory: a ``datasets.DatasetDict`` with each split.
        """
        output_dataset_dict = DatasetDict()
        for batch in DatasetDicts._dict_batch_loader(input_dict, batch_size=batch_size):
            with BoundProcessPoolExecutor(qsize=num_proc, max_workers=num_proc) as executor:
                future_dict = {}
                for split, dataset in batch:
                    if function:
                        future_dict[executor.submit(dataset_func, dataset, function, **kwargs)] = split
                    else:
                        future_dict[executor.submit(dataset_func, dataset)] = split
                for future in as_completed(future_dict):
                    try:
                        split = future_dict[future]
                        result = future.result()
                        output_dataset_dict[split] = result
                    except Exception as e:
                        traceback.print_exc(file=sys.stdout)
                        raise RuntimeError(e)
        return output_dataset_dict

    @staticmethod
    def _dataset_map(
            dataset: Union[Dataset, DatasetDict],
            function: Callable,
            **kwargs: Optional[dict]
    ) -> Union[Dataset, DatasetDict]:
        return dataset.map(
            function,
            batch_size=0,
            remove_columns=dataset.column_names,
            fn_kwargs=kwargs,
            batched=True,
        )

    def map_parallel(self, function: Callable, num_proc, batch_size, **kwargs) -> "DatasetDicts":
        self._check_values_type()
        return DatasetDicts(
            {
                name: self._process_executor(self._dataset_map, dataset_dict, function, num_proc, batch_size, **kwargs)
                for name, dataset_dict in self.items()
            }
        )

    @staticmethod
    def _parallel_handler(function: Callable[..., DatasetDict], file_dicts: Dict[str, Any]) -> "DatasetDicts":
        return DatasetDicts(
            {
                name: DatasetDicts._process_executor(function, file_dict)
                for name, file_dict in file_dicts.items()
            }
        )

    @staticmethod
    def from_csv_parallel(file_dicts) -> "DatasetDicts":
        return DatasetDicts._parallel_handler(DatasetDict.from_csv, file_dicts)

    @staticmethod
    def from_json_parallel(file_dicts) -> "DatasetDicts":
        return DatasetDicts._parallel_handler(DatasetDict.from_json, file_dicts)

    @staticmethod
    def from_parquet_parallel(file_dicts) -> "DatasetDicts":
        return DatasetDicts._parallel_handler(DatasetDict.from_parquet, file_dicts)

    @staticmethod
    def from_text_parallel(file_dicts) -> "DatasetDicts":
        return DatasetDicts._parallel_handler(DatasetDict.from_text, file_dicts)

    def to_dataset(self):
        ...

    def flatten_to_dataset_dict(self, axis: int = 0) -> DatasetDict:
        assert axis in [0, 1]
        dataset_dicts = copy.deepcopy(self)
        flatten = DatasetDict()
        if axis == 0:
            for dataset_dict_name, dataset_dict in dataset_dicts.items():
                for dataset_name, dataset in dataset_dict.items():
                    flatten_path = Path(dataset_dict_name, dataset_name).as_posix()
                    flatten[flatten_path] = dataset
        elif axis == 1:
            for dataset_dict_name, dataset_dict in dataset_dicts.items():
                dataset = [dataset for dataset in dataset_dict.values()]
                concat_dataset = concatenate_datasets(dataset, axis=0)
                flatten[dataset_dict_name] = concat_dataset
        # flatten._check_values_type()
        # flatten._check_values_features()
        return flatten

    def flatten_as_dataset(self):
        ...

    def flatten_as_dataset_dict(self):
        ...
