from dataclasses import dataclass

@dataclass
class Range:
    min: int
    max: int
    step: int

@dataclass
class Threshold:
    min: float
    max: float

@dataclass
class InputParameters:
    anomaly_multiplier: float
    forecast_roll_window: int
    diurnal_xticks: Range
    default_temp_threshold: Threshold
    default_hum_threshold: Threshold
    min_points_for_forecast: int