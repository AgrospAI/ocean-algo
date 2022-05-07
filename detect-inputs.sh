#!/bin/bash

jq -r '.[]' <(echo $DIDS) | while read i; do

  unzip -j /data/inputs/$i/0 -d /data/input/ #UNZIP INPUT DATA to /data/input/

done

wait

python detect.py --weights /model/best.pt --source /data/input/ # run yolov5 python to detect images at /data/input using trained model best.pt

wait

cd /usr/src/app/runs/detect/ # Change workdir

wait

zip -r results.zip exp*/* # Zip the results

wait

mv results.zip /data/outputs/. # Move results to /data/outputs/. to get them uploaded

wait
