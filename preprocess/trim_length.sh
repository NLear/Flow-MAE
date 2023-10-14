#! /bin/bash
src="$1"
dst="$2"
bytes="$3"

trim_packet_length() {
  #  echo triming "$1" to "$2"
  editcap -F pcap -s "$3" "$1" "$2" >>"trim_packet_length.log" 2>&1
}

if test ! -f "$src"; then
  echo "source file $src not exist"
  exit 1
fi

trim_packet_length "$src" "$dst" "$bytes"
