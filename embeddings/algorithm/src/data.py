from dataclasses import asdict, dataclass


@dataclass
class InputParameters:
    token: str
    embedding_url: str
    models_url: str | None
    model: str


@dataclass(frozen=True)
class Result:
    embeddings: list[int]
    metadata: dict[str, any]

    asdict = asdict


@dataclass
class ModelSpecification:
    name: str
    url: str
