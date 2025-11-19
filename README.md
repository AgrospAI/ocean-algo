# OceanProtocol Algorithms repository

This repository contains a set of algorithms to use inside the Ocean Protocol ecosystem, this file provides a general guideline for the implementation of a new one.

## New algorithm implementation

To develop a new algorithm, head to the `_base` directory and choose a subdirectory with the programming language that you'll use, we recommend using `python`.

```
_base
├── bash
│   ├── algorithm
│   │   ├── Dockerfile
│   │   ├── README.md
│   │   ├── run.sh
│   │   └── test.sh
│   ├── docker-compose.yaml
│   └── entrypoint.sh
└── python
    ├── Dockerfile
    ├── algorithm
    │   ├── LICENSE
    │   ├── README.md
    │   ├── pyproject.toml
    │   ├── src
    │   │   ├── __init__.py
    │   │   ├── algorithm.py
    │   │   └── data.py
    │   ├── tests
    │   │   ├── __init__.py
    │   │   └── test_main.py
    │   └── uv.lock
    └── docker-compose.yaml

```

## Running the Algorithm

To run and test an algorithm that comes from one of our templates, run the `docker-compose.yaml` file using `docker-compose up --build`. This file, configures the run to be as similar as the one it will run on in our provider.

To run in normal mode (not test), comment out the entrypoint line in the `docker-compose.yaml` file.

The upload, test and fix iteration is quite slow, so we recommend that, even locally, you keep the template's configured mounted volumes as they are, matching the final environment directory structure.

## Publishing your Asset to AgrospAI

Foremost, you need to obtain your Gaia-X Compliance credentials to gain access to the marketplace and all other Gaia-X services. To do so, follow the steps provided in [Agrospai's Incorporation Guide](https://agrospai.udl.cat/docs/guides/incorporation) or contact us via email at *agrospai@udl.cat*.

While you are waiting for your credentials, for the next step you will need to have your code or data in a public docker registry, API, GraphQL or IPFS; as of _13-10-2025_. In this repository, under the `_examples` directory, you can see a few examples on how to containerize and publish an algorithm or, in short, just use the following command. (assuming that you want to publish your algorithm as a public docker in dockerhub).

```bash
$ docker buildx build --platform linux/amd64,linux/arm64 -t {ALGORITHM_TAG}:{ALGORITHM_VERSION} . --push
```

Once you have your credentials, to upload an asset to AgrospAI's marketplace, you can do it manually via the marketplace's UI at [Portal AgrospAI](https://portal.agrospai.udl.cat) or programmatically (recommended) using the [PontusX-cli](https://github.com/AgrospAI/pontus-x_cli), which provides documentation and guideline on how to do so.

When publishing an _algorithm_, the entrypoint should point to the `entrypoint.py` file located in this repository, as in the following example publishing a YAML with `pontus-x_cli`:

```yaml
---
service:
  serviceType: compute
  fileType: url
  serviceEndpoint: "https://provider.pontus-x.eu"
  timeout: 0
  files:
    - method: GET
      type: url
      url: >-
        https://raw.githubusercontent.com/AgrospAI/ocean-algo/refs/heads/v2-ocean-runner/entrypoint.py
  pricing:
    type: free
  datatoken:
    name: [YOUR_ALGORITHM_NAME]
    symbol: [YOUR_ALGORITHM_SYMBOL]
```
