# Algorithm implementation based on the OceanProtocol ecosystem

Build and push

```bash
$ docker buildx build --platform linux/amd64,linux/arm64 -t {ALGORITHM_TAG}:{ALGORITHM_VERSION} . --push
```

_Algorithm details_

## Publish to registry

```bash
docker buildx build --platform linux/amd64,linux/arm64 -t registry.agrospai.udl.cat/library/ocean-algo-debugger:latest . --push
```
