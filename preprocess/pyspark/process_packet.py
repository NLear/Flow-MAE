from typing import Tuple, Union

from preprocess import Ether, IP, TCP, UDP, ARP, DNS, Padding, raw


def should_omit_packet(packet):
    # if len(bytes(packet)) < 800:
    #     return True

    # SYN, ACK or FIN flags set to 1 and no payload
    if TCP in packet and (packet.flags & 0x13):
        # not payload or contains only padding
        layers = packet[TCP].payload.layers()
        if not layers or (Padding in layers and len(layers) == 1):
            return True

    if UDP in packet and not packet[UDP].payload:
        return True

    # DNS segment
    if DNS in packet or ARP in packet:
        return True

    return False


def remove_ether_header(packet):
    if Ether in packet:
        return packet[Ether].payload

    return packet


def mask_ip(packet):
    if IP in packet:
        packet[IP].src = "0.0.0.0"
        packet[IP].dst = "0.0.0.0"

    return packet


def mask_udp(packet):
    if UDP in packet:
        packet[UDP].sport = 0
        packet[UDP].dport = 0

    return packet


def mask_tcp(packet):
    if TCP in packet:
        packet[TCP].sport = 0
        packet[TCP].dport = 0

    return packet


def pad_udp(packet):
    if UDP in packet:
        # get layers after udp
        layer_after = packet[UDP].payload.copy()

        # build a padding layer
        pad = Padding()
        pad.load = "\x00" * 12

        layer_before = packet.copy()
        layer_before[UDP].remove_payload()
        packet = layer_before / pad / layer_after

        return packet

    return packet


def crop_and_pad(packet, max_length=1024) -> Tuple[bytes, int]:
    packet_bytes = bytearray(raw(packet))
    origin_len = len(packet_bytes)

    if origin_len < max_length:
        packet_bytes.extend(b'\x00' * (max_length - origin_len))
    elif origin_len > max_length:
        packet_bytes = packet_bytes[:max_length]

    return bytes(packet_bytes), min(origin_len, max_length)


def transform_packet(packet) -> Union[Tuple[bytes, int], None]:
    if should_omit_packet(packet):
        return None

    packet = remove_ether_header(packet)
    # packet = pad_udp(packet)
    packet = mask_ip(packet)
    packet = mask_udp(packet)
    packet = mask_tcp(packet)

    return crop_and_pad(packet)

# def write_batch(output_path, batch_index, rows):
#     part_output_path = Path(
#         str(output_path.absolute()) + f"_part_{batch_index:04d}.json.gz"
#     )
#     with gzip.open(part_output_path, "wt") as f_out:
#         for row in rows:
#             f_out.write(f"{json.dumps(row)}\n")
# def transform_pcap(
#         pcap_path: Union[Path, str] = None,
#         target_dir_path: Union[Path, str] = None,
#         label_id: int = None,
#         output_batch_size: int = 10000,
#         max_batch: int = 1,
# ):
#     # Convert pcap_path and target_dir_path to Path objects if they are not already
#     pcap_path = Path(pcap_path)
#     target_dir_path = Path(target_dir_path)
#
#     output_path = target_dir_path / (pcap_path.name + ".transformed")
#     print("Processing", pcap_path)
#
#     rows = []
#     batch_index = 0
#     for packet in PcapReader(str(pcap_path)):
#         arr, feature_len = transform_packet(packet)
#         if arr is not None:
#             # get labels
#             row = {
#                 "label": label_id,
#                 "feature": arr.tolist(),
#                 "feature_len": feature_len
#             }
#             rows.append(row)
#
#             # write every batch_size packets, by default 10000
#             if rows and len(rows) == output_batch_size:
#                 write_batch(output_path, batch_index, rows)
#                 batch_index += 1
#                 rows.clear()
#
#             if max_batch and batch_index >= max_batch:
#                 break
#
#     # final write
#     if rows:
#         write_batch(output_path, batch_index, rows)
#
#     print(output_path, "Done")

# import itertools
# import pyarrow as pa
# from pyarrow.parquet import ParquetWriter
# from scapy.utils import PcapReader


# def transform_pcap(
#         pcap_path: Union[Path, str] = None,
#         target_dir_path: Union[Path, str] = None,
#         label_id: int = None,
#         output_batch_size: int = 10000,
#         max_batch: int = 1,
# ):
#     # Convert pcap_path and target_dir_path to Path objects if they are not already
#     pcap_path = Path(pcap_path)
#     target_dir_path = Path(target_dir_path)
#
#     output_path = target_dir_path / (pcap_path.name + ".transformed")
#     print("Processing", pcap_path)
#
#     batch_index = 0
#
#     # Define the schema for the Parquet file
#     schema = pa.schema([
#         ('label', pa.int32()),
#         ('feature', pa.list_(pa.uint8())),
#         ('feature_len', pa.int32()),
#     ])
#
#     with PcapReader(str(pcap_path)) as pcap_reader:
#         while True:
#             # Read a batch of packets using itertools.islice()
#             packet_batch = list(itertools.islice(pcap_reader, output_batch_size))
#
#             if not packet_batch:
#                 break
#
#             # Process the packet batch and create lists to hold the labels, features, and feature lengths
#             labels = []
#             features = []
#             feature_lens = []
#
#             for packet in packet_batch:
#                 feature, feature_len = transform_packet(packet)
#                 if feature is not None:
#                     labels.append(label_id)
#                     features.append(feature)
#                     feature_lens.append(feature_len)
#
#             # Create a pa.RecordBatch directly without using pd.DataFrame
#             record_batch = pa.RecordBatch.from_arrays([
#                 pa.array(labels, type=pa.int32()),
#                 pa.array(features, type=pa.list_(pa.uint8())),
#                 pa.array(feature_lens, type=pa.int32()),
#             ], schema=schema)
#
#             # Write the record batch to a Parquet file
#             part_output_path = output_path.with_name(f"{output_path.name}_part_{batch_index:04d}.parquet")
#             with ParquetWriter(part_output_path, schema, compression='snappy') as writer:
#                 writer.write_table(pa.Table.from_batches([record_batch]))
#
#             batch_index += 1
#             if max_batch and batch_index >= max_batch:
#                 break
#
#     print(output_path, "Done")
