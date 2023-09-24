#!/bin/sh

DIDS=$(echo $DIDS | sed -e "s/\"//g" | sed -e "s/\[//g" | sed -e "s/\]//g")
i=0
for DID in $(IFS=',';echo $DIDS);
do
  echo "Copying $DID to /$i.csv"
  cp /data/inputs/$DID/0 /$i.csv
  i=$((i+1))
done

wait

echo "RML mapping using $1..."
java -jar /rmlmapper.jar -m $1 -o /output.nt

wait

echo "Compressing output.nt to output.nt.gz"
gzip -c /output.nt > /data/outputs/output.nt.gz
