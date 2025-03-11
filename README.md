# OceanProtocol Algorithms repository

This repository contains a set of algorithms to use inside the Ocean Protocol ecosystem, this file provides a general guideline for the implementation of a new one.

## New algorithm implementation

To develop a new algorithm, head to the `_base` directory and choose a subdirectory with the programming language that you'll use, we recommend using `python` or `python-monolith`.

```PowerShell
└───_base
    ├───bash
    │   │   docker-compose.yaml
    │   │   entrypoint.sh
    │   │   
    │   └───algorithm
    │           README.md
    │           Dockerfile
    │           test.sh
    │           run.sh
    │
    └───python
        │   .dockerignore
        │   docker-compose.yaml
        │   Dockerfile
        │   entrypoint.sh
        │
        └───algorithm
            │   .gitignore
            │   LICENSE
            │   poetry.lock
            │   pyproject.toml
            │   README.md
            │
            ├───src
            │   │   main.py
            │   │   __init__.py
            │   │
            │   └───implementation
            │           algorithm.py
            │           __init__.py
            │
            └───tests
                    test_main.py
                    __init__.py
    
```

Copy the chosen programming language directory in the root path of the repository — with the rest of the implemented algorithms — and follow the instructions in the `README.md` file inside the copied directory. If you want to use another programming language or structure, do it, but keep in mind to test it thoroughly.

To test the algorithm, run the docker-compose inside the copied directory with the `TEST` environment variable set. For testing purposes, there is also a `_data` directory with some mocks of what will be the used directory structure in the Ocean Protocol environment, test your algorithm with them before uploading it to the blockchain to ensure that it will (most likely) run in the first attempt.

The upload, test and fix iteration is quite slow, so we recommend that, even locally, you configure the docker-compose with the proper mounted volumes so you only need to build the project when changing dependecies.

```bash
$ docker compose up --build
```

## Upload your asset to AgrospAI

Foremost, you need to obtain your Gaia-X Compliance credentials to gain access to the marketplace and all other Gaia-X services. To do so, contact us via email.

To upload your asset to AgrospAI's marketplace. you can do so programmatically with the `pontus-x_cli` library, which you can install with `npm` or any other JavaScript package manager. To do so, follow the instructions at [PontusX CLI](https://github.com/AgrospAI/pontus-x_cli), alongside with the given examples.

Alternatively you need to follow the instructions in the *<ins>PUBLISH</ins>* section of the header.