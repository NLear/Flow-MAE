from multiprocessing import Queue
from pathlib import Path

from preprocess.factory import FactoryMeta


class BaseAdaptor(metaclass=FactoryMeta):
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def preprocess_function(self, packet, label):
        raise NotImplementedError("Subclasses should implement this method")

    def transform_pcap(self, packet_queue: Queue, num_producers: int, output_path: Path):
        raise NotImplementedError("Subclasses should implement this method")

    def __call__(self, *args, **kwargs):
        return self.transform_pcap(*self.args, **self.kwargs)


# Now, we should update the transform_pcap_factory to use the registry from BaseAdaptor
class AdaptorFactory:
    def __init__(self):
        self._creators = BaseAdaptor.registry

    def create(self, key, *args, **kwargs):
        creator = self._creators.get(key.lower())
        if not creator:
            raise ValueError(f"No registered transform pcap class for '{key}'")
        return creator(*args, **kwargs)


transform_pcap_factory = AdaptorFactory()
