# OceanProtocol Algorithms repository

This repository contains a set of algorithms to use inside the Ocean Protocol ecosystem, this file provides a general guideline for the implementation of a new one.

## New algorithm implementation

To develop a new algorithm, head to the `_base` directory and choose a subdirectory with the programming language that you'll use, we recommend using Python.

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

To test the algorithm, run the docker-compose inside the copied directory with the `TEST` environment variable set to a truthy value. For testing purposes, there is also a `_data` directory with some mocks of what will be the used directory structure in the Ocean Protocol environment, test your algorithm with them before uploading it to the blockchain to ensure that it will (most likely) run in the first attempt.

```bash
# changing the environment value inside the docker-compose.yaml
$ docker compose up --build
```

## Upload to AgrospAI

! TODO: 