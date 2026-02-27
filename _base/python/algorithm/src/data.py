from pydantic import BaseModel


class InputParameters(BaseModel):
    age: int
    name: str
