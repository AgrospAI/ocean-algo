from typing import Any
from pydantic import BaseModel, Field, field_validator

class InputParameters(BaseModel):
    province: str
    municipality: str
    aggregation: str = Field(default="0")
    zone: str = Field(default="0")
    polygon: str
    parcel: str
    precint: str

    @field_validator("province", "municipality", "aggregation", "zone", "polygon", "parcel", "precint", mode="before")
    @classmethod
    def must_be_numeric_positive(cls, v: Any) -> str | None:
        if v is None:
            return v
        stripped = str(v).strip()
        if stripped and not stripped.isdigit():
            raise ValueError(f"'{stripped}' must be a positive numeric value")
        return stripped


    def to_cql2_filter(self) -> str:
        parts = [
            f'provincia={self.province}',
            f'municipio={self.municipality}',
            f'agregado={self.aggregation}',
            f'zona={self.zone}',
            f'poligono={self.polygon}',
            f'parcela={self.parcel}',
            f'recinto={self.precint}',
        ]
        return " AND ".join(parts)


    def to_sigpac_params(self) -> dict[str, Any]:
        return {
            'f': 'json',
            'limit': 1,
            'filter': self.to_cql2_filter(),
        }