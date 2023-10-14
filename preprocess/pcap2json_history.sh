#! /bin/bash
src="$1"
dst="$2".txt
packet_num="$3"

#tcp2txt() {
##  echo writing "$2"
#  tshark -T ek -r "$1" -c "$3" -Q >"$2"  2>/dev/null
##  tshark -T fields -r "$1" -c "$3" -e "tcp.payload" -Q >"$2" 2>/dev/null
#}
#
#udp2txt() {
##  echo writing "$2"
#  tshark -T ek -r "$1" -c "$3" -Q >"$2"  2>/dev/null
##  tshark -T fields -r "$1" -c "$3" -e "udp.payload" -Q >"$2" 2>/dev/null
#}

if test ! -f "$src"; then
  echo "source file $src not exist"
  exit 1
fi

#file_basename=$(basename "$src")
#
#if [[ $file_basename =~ ^.*_([[:alpha:]]{3}).pcap.HostPair_ ]]; then
#  if [[ ${BASH_REMATCH[1]} == "tcp" ]]; then
#    tcp2txt "$src" "$dst" "$packet_num"
#  elif [[ ${BASH_REMATCH[1]} == "udp" ]]; then
#    udp2txt "$src" "$dst" "$packet_num"
#  else
#    echo "no match protocol"
#  fi
#else
#  echo "regex comand failed"
#fi


 #!/bin/sh

#  tshark -T fields -e "frame.number" -e "frame.time_epoch" -e "frame.len" -e "frame.cap_len" -e "frame.protocols" \
#  -e "tcp.payload"  \
#  -r  ./capDESKTOP-AN3U28N-172.31.64.17_tcp.pcap.HostPair_5-101-40-105_172-31-64-17.pcap  > out.json

#tshark -T ek -r "$1" -c 16 -j "udp.payload" -Q >"$2"  2>/dev/null

# tshark -T json -e "frame.number" -e "frame.time_epoch" -e "frame.time_delta" -e "frame.time_relative" \
# -e "frame.len" -e "frame.cap_len" -e "frame.protocols" \
# -e "ip_ip_version" "ip_ip_hdr_len" "ip_ip_len"  \
# -e "tcp.payload"  \
# -r  ./capDESKTOP-AN3U28N-172.31.64.17_tcp.pcap.HostPair_5-101-40-105_172-31-64-17.pcap  > result.json