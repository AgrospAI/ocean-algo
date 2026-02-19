from __future__ import annotations
from dataclasses import dataclass

@dataclass
class InputParameters:
    model_image_digest_reference: str | None = None
    
