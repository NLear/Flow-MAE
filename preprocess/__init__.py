from scapy.compat import raw
from scapy.layers.dns import DNS
from scapy.layers.inet import IP, TCP, UDP
from scapy.layers.l2 import Ether, ARP
from scapy.packet import Raw, Padding
from scapy.utils import wrpcap, rdpcap, PcapWriter, PcapReader



FEATURE_COL = "x"
LABEL_COL = "labels"
FEATURE_LEN_COL = "feature_len"
FEATURE_LEN_MAX = 1024

DATASETS_FILE = "dataset_dict"
LABEL_MAPPING_FILE = "label_mapping.json"


__all__ = [
    "Ether",
    "IP",
    "TCP",
    "UDP",
    "Raw",
    "wrpcap",
    "rdpcap",
    "PcapWriter",
    "PcapReader",
    "ARP",
    "DNS",
    "Padding",
    "raw",

    "FEATURE_COL",
    "LABEL_COL",
    "FEATURE_LEN_COL",
    "FEATURE_LEN_MAX",

    "DATASETS_FILE",
    "LABEL_MAPPING_FILE",
]