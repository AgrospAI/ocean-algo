from pydantic import BaseModel


class InputParameters(BaseModel):
    aggregate_api: str
    csv_separator: str = ";"
