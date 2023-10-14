#! /bin/bash
tshark -r "$1" -T fields -e frame.number -e frame.time_relative -e frame.protocols \
-e ip.src -e ip.dst -e ip.proto -e tcp.srcport -e tcp.dstport  -e frame.len -e _ws.col.Info -e tcp.payload \
-E header=y -E separator=/t -E quote=n -E occurrence=f > "$2"
