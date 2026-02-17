from pydantic import BaseModel


class AggregateAPI(BaseModel):
    url: str


class InputParameters(BaseModel):
    aggregate_api: AggregateAPI
    csv_separator: str = ";"
