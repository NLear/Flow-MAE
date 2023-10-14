#! /bin/bash
src="$1"
min_packet_num="$2"

rm_small_pcap() {
  if test ! -f "$1"; then
    echo "source file $1 not exist"
    exit 1
  fi

  num=$(capinfos -c -M "$1" | grep -Po "^Number of packets:\s*[\d]+$" | grep -Po "[\d]+$")
#  echo Number of packets: "${num}"
  if [[ "${num}" -le "$2" ]]; then
    rm "$1"
#    echo removing small file "$1", packets: "${num}"
  fi
}

rm_small_pcap "${src}" "$min_packet_num"
