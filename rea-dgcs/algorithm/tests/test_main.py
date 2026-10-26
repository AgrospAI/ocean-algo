import os
import tempfile
from pathlib import Path
from src.algorithm import algorithm


def test_main():
    data_dir = Path(__file__).resolve().parent.parent.parent / "_data"
    algorithm.configuration.environment.base_dir = data_dir
    algorithm()

    assert os.listdir(algorithm.job_details.paths.outputs)


def test_ocean_node_42_layout():
    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        (base / "inputs").mkdir()
        (base / "ddos").mkdir()  # empty ddos directory in Ocean Node 4.2
        (base / "outputs").mkdir()
        (base / "logs").mkdir()

        sample_src = (
            Path(__file__).resolve().parent.parent.parent
            / "_data/inputs/eb60f87363a36a5ae5cb8373524a8fd976b0cc5f8c40a706c615b857ae0e2974/0"
        )
        (base / "inputs" / "input_rea.json").write_bytes(sample_src.read_bytes())

        algorithm.configuration.environment.base_dir = base
        algorithm()

        outputs = os.listdir(algorithm.job_details.paths.outputs)
        assert "parcel_map.html" in outputs
        assert "parcel_log.json" in outputs


