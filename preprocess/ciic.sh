#! /bin/bash
src="$1"
dst="$2"

getdir() {
  echo "traverse dir" $1
  for file in "$1"/*
  do
  if test -f "$1/${file}"
  then
    if [[ "${file##*.}" == "pcap" ]]; then
      echo "cicflowmeter processing" "$file"
      cicflowmeter -f "$1/${file}" -c "$1/${file%.*}.csv"
    fi
  else
      getdir "$file"
  fi
  done
}

getdir "${src}"
