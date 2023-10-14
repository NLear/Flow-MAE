#! /bin/bash
src="$1"
dst="$2"
time="$3"

trim_pcap() {
#  echo triming "$1" to "$2"
  editcap -F pcap -i "$3" -s "$4" "$1" "$2"
}

if test ! -f "$src"; then
  echo "source file $src not exist"
  exit 1
fi

file_basename=$(basename "$src")
if [[ $file_basename =~ ^.*_([[:alpha:]]{3}).pcap.HostPair_ ]]; then
  if [[ ${BASH_REMATCH[1]} == "tcp" ]]; then
    editcap -F pcap -i "$time" "$src" "$dst"
  elif [[ ${BASH_REMATCH[1]} == "udp" ]]; then
    trim_pcap "$src" "$dst" "$time" 170
  else
    echo "no match protocol"
  fi
else
  echo "regex comand failed"
fi
