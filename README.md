# OceanProtocol Algorithms repository

This repository contains a set of algorithms to use inside the Ocean Protocol ecosystem, this file provides a general guideline for the implementation of a new one.

## New algorithm implementation

To develop a new algorithm, head to the `_base` directory and choose a subdirectory with the programming language that you'll use, we recommend using Python (right now it's the only one developed).

```PowerShell
└───_base
    └───python
        └───algorithm
            ├───src
            │   └───implementation
            └───tests
```

Copy the chosen programming language directory in the root path of the repository – with the rest of the implemented algorithms — and follow the instructions in the `README.md` file inside the copied directory.

To test the algorithm, run the docker-compose inside the copied directory with the `TEST` environment variable set to a truthy value. For testing purposes, there is also a `_data` directory with some mocks of what will be the used directory structure in the Ocean Protocol environment, test your algorithm with them before uploading it to the blockchain to ensure that it will (most likely) run in the first attempt.

```bash
# changing the environment value inside the docker-compose.yaml
$ docker compose up --build
```

## Upload to AgrospAI

! TODO: 