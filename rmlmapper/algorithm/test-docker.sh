docker build -t rogargon/rml-mapper .
docker run \
-v "$(pwd)"/data/ddos/:/data/ddos/ \
-v "$(pwd)"/data/inputs/:/data/inputs/ \
-v "$(pwd)"/data/outputs/:/data/outputs/ \
-v "$(pwd)"/weight-map.ttl:/weight-map.ttl \
-v "$(pwd)"/feed-map.ttl:/feed-map.ttl \
-e DIDS='["8f67E08be5dD941a701c2491E814535522c33bC2"]' \
-e ALGO='/feed-map.ttl' \
rogargon/rml-mapper /map-inputs.sh /feed-map.ttl
