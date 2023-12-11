#!/bin/bash

i=0
jq -r '.[]' <(echo $DIDS) | while read DID; do
  echo "Copying $DID to /$i.csv"
  cp /data/inputs/$DID/0 /$i.csv
  i=$((i+1))
done

wait

cat /data/inputs/algoCustomData.json
MAPPING=$(cat /data/inputs/algoCustomData.json | jq -r '.mapping')
if [[ $MAPPING == *"null"* ]]; then
  echo "Mapping not found in algoCustomData.json, using default"
  MAPPING=$(echo "https://raw.githubusercontent.com/rogargon/ocean-algo/rmlpooler/feed-map.ttl")
fi
echo "RML mapping using $MAPPING"
java -jar /rmlmapper.jar -m $MAPPING -o output.nt
echo "Generated $(wc -l <output.nt) triples"

wait

DID=$(jq -r '.[0]' <(echo $DIDS))
echo "Pooling mapped data for $DID..."
curl -s -X POST "http://kg.dataspace:3030/ds?graph=did%3Aop%3A$DID" \
     -H "Content-Type: application/n-triples" \
     -u admin:password \
     --data-binary @output.nt > /data/outputs/pooling.log
cat /data/outputs/pooling.log
