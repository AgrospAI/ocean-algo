from dataclasses import dataclass
from typing import List

from pydantic import BaseModel


class InputParameters(BaseModel):
    token: str
    embedding_url: str
    models_url: str | None = None
    model: str | None = None
    timeout: int = 60


@dataclass(frozen=True)
class Metadata:
    filepath: str
    did: str
    idx: str


class Result(BaseModel):
    embeddings: List[List[float]]
    metadata: Metadata


class ModelSpecification(BaseModel):
    name: str
    url: str
