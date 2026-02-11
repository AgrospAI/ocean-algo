from pathlib import Path
from typing import Tuple

import jinja2
import pandas as pd  # type: ignore

from src.aggregate import Aggregate

from .config_schema import DIMENSION_LABELS
from .preprocessing import (
    calculate_maturity_kpis,
    compare_company_to_aggregate,
    process_survey,
)


def get_template(
    base_path: Path, template_name: str = "template.j2"
) -> jinja2.Template:
    template_loader = jinja2.FileSystemLoader(searchpath=base_path)
    template_env = jinja2.Environment(loader=template_loader)

    return template_env.get_template(template_name)


def benchmark(
    aggregate: Aggregate,
    aggregate_filter: str,
    survey: pd.DataFrame,
) -> Tuple[dict, str]:
    survey = process_survey(survey)
    survey = calculate_maturity_kpis(survey)

    comparison = compare_company_to_aggregate(
        survey.iloc[0], aggregate[aggregate_filter]
    )

    base_path = Path("src") / "benchmarking"

    translations = base_path / "translations.json"
    template = get_template(base_path)

    return (
        comparison,
        template.render(
            **comparison,
            translations=translations.read_text(),
            dimension_labels=DIMENSION_LABELS,
        ),
    )
