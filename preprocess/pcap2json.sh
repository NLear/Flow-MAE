#! /bin/bash

src="$1"
dst="$2"
min_packet_num="$3"
min_file_size="$4"

if test ! -f "$src"; then
  echo "source file $src not exist"
  exit 1
fi

pcap2csv() {
  tshark -T fields -e frame.number -e frame.time_relative \
    -e tcp.stream -e ip.src -e tcp.srcport -e ip.dst -e tcp.dstport \
    -e frame.len -e ip.len -e tcp.payload \
    -E header=y -E separator=, -E occurrence=f \
    -Y "ip and tcp.payload" \
    -r "$src" >"$dst"_TCP.csv

  tshark -T fields -e frame.number -e frame.time_relative \
    -e udp.stream -e ip.src -e udp.srcport -e ip.dst -e udp.dstport \
    -e frame.len -e ip.len -e udp.payload \
    -E header=y -E separator=, -E occurrence=f \
    -Y "ip and udp.payload" \
    -r "$src" >"$dst"_UDP.csv
}

file_size=$(stat --format=%s "$src")
if [ "$file_size" -ge "$min_file_size" ]; then
  num=$(capinfos -c -M "$src" | grep -Po "^Number of packets:\s*[\d]+$" | grep -Po "[\d]+$")
  if [ "${num}" -ge "$min_packet_num" ]; then
    pcap2csv
  fi
fi
