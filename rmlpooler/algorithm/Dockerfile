FROM rmlio/rmlmapper-java

RUN apt-get update && apt-get install -y curl jq \
  && rm -rf /var/lib/apt/lists/*
WORKDIR /
ENTRYPOINT ["/bin/bash"]
