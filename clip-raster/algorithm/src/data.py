from pydantic import BaseModel, ConfigDict, Field


class InputParameters(BaseModel):
    """Algorithm parameters, read from `algoCustomData.json`."""

    model_config = ConfigDict(populate_by_name=True)

    lookback_days: int = Field(default=60, ge=1, le=365, alias="lookbackDays")
    """How many days back to search for Sentinel-2 scenes."""

    max_cloud_cover: float = Field(default=20, ge=0, le=100, alias="maxCloudCover")
    """Maximum share (%) of cloudy, shadowed or missing pixels inside the parcel."""
