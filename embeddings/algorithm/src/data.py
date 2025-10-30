from dataclasses import asdict, dataclass


@dataclass
class InputParameters:
    token: str
    embedding_url: str
    is_zipped: bool = False
    models_url: str | None = None
    model: str | None = None


@dataclass(frozen=True)
class Result:
    embeddings: list[int]
    metadata: dict[str, any]

    asdict = asdict


@dataclass
class ModelSpecification:
    name: str
    url: str
