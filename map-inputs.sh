#!/bin/sh

DIDS=$(echo $DIDS | sed -e "s/\"//g" | sed -e "s/\[//g" | sed -e "s/\]//g")
i=0
for DID in $(IFS=',';echo $DIDS);
do
  echo "Copying $DID to /data/$i.csv"
  cp /data/inputs/$DID/0 /data/$i.csv
  i=$((i+1))
done

wait

echo "RML mapping..."
java -jar /rmlmapper.jar -m /data/map.ttl -o /data/output.nt

wait

echo "Compressing output and copying to output.nt.gz"
gzip -c /data/output.nt > /data/outputs/output.nt.gz
