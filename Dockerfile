FROM rmlio/rmlmapper-java

COPY map.ttl /map.ttl
COPY map-inputs.sh /map-inputs.sh
WORKDIR /

ENTRYPOINT ["/bin/sh"]
