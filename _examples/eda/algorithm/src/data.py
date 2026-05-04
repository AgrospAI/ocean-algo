from pydantic import BaseModel, Field


class InputParameters(BaseModel):
    title: str = Field("Profiling Report")
    sensitive: bool = Field(False)
