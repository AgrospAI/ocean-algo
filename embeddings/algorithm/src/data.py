from dataclasses import dataclass, asdict


@dataclass
class InputParameters:
    token: str
    model_url: str
    model: str


@dataclass(frozen=True)
class Result:
    embeddings: list[int]
    metadata: dict[str, any]

    asdict = asdict
