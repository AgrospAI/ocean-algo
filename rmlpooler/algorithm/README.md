## RML Mapper Algorithm

Based on [YARRRML](https://rml.io/yarrrml/) mapping definition files, for instance `feed-map.yml` and `weight-map.yml`.

YARRRML mappings are easier to generate and edit, thanks to a more compact syntax. They can be converted into the
standard RML syntax using the [YARRRML Parser](https://hub.docker.com/r/rmlio/yarrrml-parser) Docker image.

```shell
docker pull rmlio/yarrrml-parser
docker run --rm -v $(pwd)/:/data rmlio/yarrrml-parser:latest -i feed-map.yml -o feed-map.ttl
```

Then the algorigthm will apply the [RML Mapper](https://hub.docker.com/r/rmlio/rmlmapper-java) to generate the RDF triples from the CSV input file.
This process can be also done locally:

```shell
docker pull rmlio/rmlmapper-java
docker run --rm -v $(pwd)/:/data rmlio/rmlmapper-java -m /data/map.ttl -o /data/out.nt
```
