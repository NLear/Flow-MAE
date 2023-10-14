#! /bin/bash
src="$1"
dst="$2"
min_packet_num="$3"
max_packet_num="$4"
packet_num="$5"

split_packets() {
  if test ! -f "$1"; then
    echo "source file $1 not exist"
    exit 1
  fi

  num=$(capinfos -c -M "$1" | grep -Po "^Number of packets:\s*[\d]+$" | grep -Po "[\d]+$")
  #  echo Number of packets: "${num}"
  if [ "${num}" -le "$3" ]; then
    rm "$1"
    #    echo removing small file "$1", packets: "${num}"
  elif [ "${num}" -gt "$4" ]; then
    split_dst="$2"_from"$num".pcap
    editcap -F pcap -c "$5" -r "$1" "$split_dst" 1-"$4"
  else
    split_dst="$2"_from"$num".pcap
    editcap -F pcap -c "$5" "$1" "$split_dst"
  fi
}

split_packets "${src}" "${dst}" "$min_packet_num" "$max_packet_num" "$packet_num"
