from preprocess.factory import FactoryMeta


class BaseAggregator(metaclass=FactoryMeta):
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def aggregator(self, source, target, min_feature_len: int = None, sample_per_label: int = None,
                   test_ratio: float = None) -> None:
        raise NotImplementedError("Subclasses should implement this method")

    def __call__(self, *args, **kwargs):
        return self.aggregator(*self.args, **self.kwargs)


# File: aggregator_factory.py
class AggregatorFactory:
    def __init__(self):
        self._creators = BaseAggregator.registry

    def create(self, key, *args, **kwargs):
        creator = self._creators.get(key.lower())
        if not creator:
            raise ValueError(f"No registered aggregator class for '{key}'")
        return creator(*args, **kwargs)


aggregator_factory = AggregatorFactory()
