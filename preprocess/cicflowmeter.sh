#! /bin/bash
src_root="${1:-.}"
dst_root="${2:-../output}"
src_record="/tmp/src.txt"
dst_record="/tmp/dst.txt"

traverse_dir() {
  echo "traverse dir $1"
  dst_dir=$dst_root${1/$src_root/}
  mkdir -p "${dst_dir}"
  for file in "$1"/*; do
    if [ -f "${file}" ] && [ "${file##*.}" = "pcap" ]; then
      base=$(basename "${file}")
      outfile="${dst_dir}/${base%.*}.csv"
      echo "processing file $file -> $outfile"
      echo "$file" >>"$src_record"
      echo "$outfile" >>"$dst_record"
      #      cicflowmeter -f "$file" -c "$outfile"
    elif [ -d "${file}" ]; then
      traverse_dir "$file"
    fi
  done
}

run_cic() {
  if [ ! -f "$1" ] || [ ! -f "$2" ]; then
    echo "$1 or $2 does not exist"
    return
  fi
  parallel --jobs 16 --linebuffer --progress  --xapply "cicflowmeter -f {1} -c {2}" :::: "$1"   :::: "$2"
}

if [ -f "$src_record" ]; then
  rm "$src_record"
fi
if [ -f "$dst_record" ]; then
  rm "$dst_record"
fi

traverse_dir "$src_root"
run_cic "$src_record" "$dst_record"
