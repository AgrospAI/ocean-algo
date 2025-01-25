from logging import getLogger
from pathlib import Path
from typing import Any, Optional

from oceanprotocol_job_details.dataclasses.job_details import JobDetails
from ultralytics import YOLO

logger = getLogger(__name__)


class Algorithm:
    def __init__(self, job_details: JobDetails):
        self._job_details = job_details
        self.results: Optional[Any] = None

    def _validate_input(self) -> "Algorithm":
        if not self._job_details.dids or len(self._job_details.dids) == 0:
            logger.warning("No DIDs found")
            raise ValueError("No DIDs found")

        if not self._job_details.files:
            logger.warning("No files found")
            raise ValueError("No files found")

    def run(self) -> "Algorithm":
        self._validate_input()

        first_did = self._job_details.dids[0]
        filename = self._job_details.files[first_did][0]

        model = YOLO('yolov8n.pt')
        model(source=filename, show=False, conf=0.4, save=True)

        output_dir = model.predictor.save_dir
        output_image_path = list(output_dir.glob('*.jpg'))[0]
        
        self.results = str(output_image_path)
        logger.info(f'Generated prediction for image {filename}')

        return self

    def save_result(self, path: Path) -> None:
        with open(self.results, 'rb') as src:
            with open(path, 'wb') as dst:
                dst.write(src.read())
