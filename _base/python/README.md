# Algorithm implementation based on the OceanProtocol ecosystem

Build and push

```bash
$ docker buildx build --platform linux/amd64,linux/arm64 -t {ALGORITHM_TAG}:{ALGORITHM_VERSION} . --push
```

# Execution

To execute the algorithm and to test it:

```bash
ocean-execute (algorithm.module.path) ([--base-dir|-b] Base data directory)

ocean-test (algorithm.module.path) ([--base-dir|-b] Base data directory) -- [pytest arguments]
```

# Implementation details

_Algorithm details_
