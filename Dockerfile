FROM rmlio/rmlmapper-java

COPY data/map.ttl /data/map.ttl
COPY map-inputs.sh /map-inputs.sh

ENTRYPOINT ["/bin/sh"]
