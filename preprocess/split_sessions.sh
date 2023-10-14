#! /bin/bash

src="$1"
dst="$2"
min_packet_num="$3"
min_file_size="$4"
split_level="$5" #hostpair, flow
split_cap_exe="$6"

if test ! -f "$src"; then
  echo "source file $src not exist"
  exit 1
fi

file_size=$(stat --format=%s "$src")
if [ "$file_size" -ge "$min_file_size" ]; then
  num=$(capinfos -c -M "$src" | grep -Po "^Number of packets:\s*[\d]+$" | grep -Po "[\d]+$")
  if [ "${num}" -ge "$min_packet_num" ]; then
    mono "$split_cap_exe" -p 1000 -b 1024000 -d -s "$split_level" -r "$src" -o "$dst"
  fi
fi
