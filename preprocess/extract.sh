#! /bin/bash
src="$1"
dst="$2"

filter_pcap() {
  if test ! -f "$1"; then
    echo "source file $1 not exist"
    exit 1
  fi
  #  echo writing "$2"
  #  tshark -r "$1" -w "$2" -2 -F pcap -R "$3" -Q >>"tshark.log" 2>&1
  tshark -r "$1" -w "$2" -F pcap -Y "$3" -Q >>"tshark.log" 2>&1
}

#rm_empty_pcap() {
#  num=$(capinfos -c "$1" | grep -Po "^Number of packets:\s*[\d,]+$" | grep -Po "[\d,]+$" | sed 's/,//g')
#  echo Number of packets: "${num}"
#  if [ "${num}" -eq 0 ]; then
#    rm "$1"
#    #    echo removing empyty file "$1"
#  fi
#}

tcp_file="${dst}"_tcp.pcap
filter_pcap "$src" "$tcp_file" "tcp.payload"
#rm_empty_pcap "$tcp_file"

udp_file="${dst}"_udp.pcap
filter_pcap "$src" "$udp_file" "udp.payload"
#rm_empty_pcap "$udp_file"
