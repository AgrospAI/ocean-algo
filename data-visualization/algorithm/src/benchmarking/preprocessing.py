import re
import unicodedata
from collections import defaultdict

import pandas as pd  # type: ignore

from .config_schema import (
    CNAE_MAP,
    KEYWORD_RULES,
    QUEST_MAPPING,
    QUESTION_REGISTRY,
    SCORING_MAPS,
    SURVEY_SCHEMA,
)


def read_parse_csv(file_path: str) -> pd.DataFrame:
    """
    CSV reader with multiple encoding support.
    """
    encodings = ["utf-8", "latin-1", "iso-8859-1", "cp1252"]

    for encoding in encodings:
        try:
            return pd.read_csv(file_path, encoding=encoding, sep=";")
        except UnicodeDecodeError:
            continue
        except Exception:
            continue

    print(f"WARNING: Could not read file {file_path}")
    return pd.DataFrame()


def normalize_questions_id(text: str) -> str | None:
    """
    Standardizes question string to create a unique key.
    """
    if pd.isna(text) or re.match(r"^\d", str(text)):
        return None

    text = str(text).lower().strip()
    text = re.sub(r"\s*\(.*?\)", "", text)
    text = unicodedata.normalize("NFD", text).encode("ascii", "ignore").decode("ascii")
    text = text.replace("/", " ").replace("-", " ").replace(".", " ")
    text = re.sub(r'"', "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", "_", text)
    return text[:50]


def normalize_free_text(text: str, rule_type: str) -> str:
    """
    Basing on question value type access to keywords dictionary and categorize them.
    """
    if not text or str(text).lower() == "nan":
        return "No especificado"

    clean_text = str(text).lower().strip()
    rules = KEYWORD_RULES.get(rule_type, {})

    for category, keywords in rules.items():
        for keyword in keywords:
            if keyword in clean_text:
                return category

    return str(text).title()


def normalize_response_value(row: pd.Series) -> str | int | float:
    """
    Normalize raw CSV answers into clean Data.
    """
    raw_val = str(row["Valor"]).strip()
    field_id = str(row.get("internal_id", ""))

    schema_options = SURVEY_SCHEMA.get(field_id, [])
    if schema_options == ["NUMERIC"]:
        digits = re.sub(r"[^\d\.,]", "", raw_val).replace(",", ".")
        try:
            # If it has decimal, return float else int
            return float(digits) if "." in digits else int(digits)
        except ValueError:
            return 0

    if "cnae" in field_id or "profile" in field_id:
        if raw_val.isdigit():
            return CNAE_MAP.get(raw_val, "Otro")
        return normalize_free_text(raw_val, rule_type="sector")

    if field_id in ["erp_in_use", "crm_in_use", "powerbi_usage"]:
        # Check if value is in predefined csv questions options if not use keyword dictionary
        if raw_val in schema_options:
            return raw_val
        return normalize_free_text(raw_val, rule_type="software")

    if "channel" in field_id or "comunicacion" in field_id:
        return normalize_free_text(raw_val, rule_type="channel")

    if "antivirus" in field_id:
        return normalize_free_text(raw_val, rule_type="antivirus")

    clean_val = raw_val.replace('"', "")
    clean_val = re.sub(
        r"([Mm]enos (de|del)|[Mm]enor que)\s+", "<", clean_val, flags=re.IGNORECASE
    )
    clean_val = re.sub(
        r"([Mm]ás (de|del)|[Mm]ayor que)\s+", ">", clean_val, flags=re.IGNORECASE
    )
    clean_val = re.sub(
        r"Entre\s+(.*?)\s+y\s+(.*)", r"\1-\2", clean_val, flags=re.IGNORECASE
    )
    clean_val = re.sub(r"\s*-\s*", "-", clean_val)

    options = schema_options
    if not options or options == ["TEXT"]:
        return clean_val.title()

    for opt in options:
        if clean_val.lower() == opt.lower():
            return opt
        opt_clean = opt.split("(")[0].strip()
        if len(opt_clean) > 3 and opt_clean.lower() in clean_val.lower():
            return opt

    return clean_val


def process_survey(df: pd.DataFrame) -> pd.DataFrame:
    assert not df.empty or "Campo" in df.columns, (
        "Invalid data, is empty or has no `Campo` column"
    )

    df["questions"] = df["Campo"].apply(normalize_questions_id)
    df = df.dropna(subset=["questions"])
    df["internal_id"] = df["questions"].map(QUEST_MAPPING).fillna(df["questions"])
    df["normalized_value"] = df.apply(normalize_response_value, axis=1)

    # Create a single row dataframe for this client
    return df[["internal_id", "normalized_value"]].set_index("internal_id").T


def calculate_maturity_kpis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates Digital Maturity Scores (0-100)
    """

    def get_score(col_name, map_type) -> float:
        """
        Retrieves score using SCORING_MAPS, score rules predefined in config_schema.
        """
        if col_name not in df.columns:
            return 0

        if map_type == "powerbi":
            # Case special for powerbi usage question, if it contains "ninguno" assign 0 else the company used powerbi
            return df[col_name].apply(
                lambda x: 0 if str(x).lower() == "ninguno" else 100
            )

        if map_type == "free_percentage":
            # For free percentage questions, we will consider the value as percentage if it's between 0 and 100, else 0
            def percentage_score(x):
                try:
                    value = float(x)
                    if 0 <= value <= 25:
                        return 25
                    elif 25 < value <= 50:
                        return 50
                    elif 50 < value <= 75:
                        return 75
                    elif 75 < value <= 100:
                        return 100
                except ValueError:
                    pass
                return 0

            return df[col_name].apply(percentage_score)

        # Select scoring maps from config schema
        mapping_dict = SCORING_MAPS.get(map_type, {})

        return df[col_name].map(mapping_dict).fillna(0)

    df["KPI_OPERATIONS"] = (
        get_score("key_processes_digitized_pct", "percentage") * 0.4
        + get_score("it_infrastructure_type", "infrastructure") * 0.3
        + get_score("collaboration_tools_usage", "binary") * 0.15
        + get_score("powerbi_usage", "powerbi") * 0.15
    )

    df["KPI_BUSINESS"] = (
        get_score("digital_revenue", "revenue") * 0.4
        + get_score("ai_for_automation_usage", "ai") * 0.20
        + get_score("active_internet_presence", "binary") * 0.20
        + get_score("digital_marketing_use", "binary") * 0.20
    )

    df["KPI_SECURITY"] = (
        get_score("two_factor_authentication", "binary") * 0.25
        + get_score("continuity_and_recovery_plans", "binary") * 0.25
        + get_score("phishing_simulations", "binary") * 0.25
        + get_score("data_protection_compliance", "binary") * 0.25
    )

    df["KPI_CULTURE"] = (
        get_score("advanced_digital_skills_pct", "percentage") * 0.25
        + get_score("cybersecurity_training", "binary") * 0.25
        + get_score("remote_work_acceptable_use_policy", "percentage") * 0.25
        + get_score("employees_using_antivirus_pct", "free_percentage") * 0.25
    )

    # Calculate dmi global score
    df["GLOBAL_SCORE"] = (
        df["KPI_OPERATIONS"] * 0.30
        + df["KPI_SECURITY"] * 0.30
        + df["KPI_BUSINESS"] * 0.20
        + df["KPI_CULTURE"] * 0.20
    ).round(1)

    # Assign maturity labels based on global score
    def assign_label(score):
        if score >= 80:
            return "Líder Digital"
        if score >= 60:
            return "Avanzado"
        if score >= 40:
            return "En Desarrollo"
        return "Principiante Digital"

    df["MATURITY_LABEL"] = df["GLOBAL_SCORE"].apply(assign_label)
    return df


def compare_kpis(company_row, aggregate_block):
    comparisons = {}

    for kpi, stats in aggregate_block["scores"].items():
        value = company_row[kpi]

        if value < stats["p25"]:
            position = "below_p25"
        elif value < stats["median"]:
            position = "between_p25_median"
        elif value < stats["p75"]:
            position = "between_median_p75"
        else:
            position = "above_p75"

        comparisons[kpi] = {
            "company": float(round(value, 1)),
            "sector_p25": stats["p25"],
            "sector_median": stats["median"],
            "sector_p75": stats["p75"],
            "position": position,
        }

    return comparisons


def compare_adoption(company_row, aggregate_block):
    comparisons = {}

    for col, adoption_pct in aggregate_block["adoption_rates"].items():
        company_value = company_row[col] == "Sí"

        comparisons[col] = {
            "company": company_value,
            "sector_adoption_pct": adoption_pct,
            "gap_pct": round((100 if company_value else 0) - adoption_pct, 1),
        }

    return comparisons


def compare_market_tools(company_row, aggregate_block):
    comparisons = {}

    for col, leaders in aggregate_block["market_leaders"].items():
        company_tool = company_row[col]

        rank = None
        share = 0.0

        for i, entry in enumerate(leaders, start=1):
            if entry["tool"] == company_tool:
                rank = i
                share = entry["share_pct"]
                break

        comparisons[col] = {
            "company": company_tool,
            "market_rank": rank,  # None = niche / unique
            "market_share_pct": share,
        }

    return comparisons


def compare_ordinal(company_row, aggregate_block):
    result = {}

    if "it_outsourcing_level" in aggregate_block["averages"]:
        result["it_outsourcing_level"] = {
            "company": company_row["it_outsourcing_level"],
            "sector_avg_score": aggregate_block["averages"]["it_outsourcing_level"][
                "avg_score"
            ],
            "sector_label": aggregate_block["averages"]["it_outsourcing_level"][
                "label"
            ],
        }

    return result


def compare_responses(company_row: pd.DataFrame, aggregate_block: dict):
    max_dim: int = max(cfg["dimension"] for cfg in QUESTION_REGISTRY.values())
    result: list = [[] for _ in range(max_dim + 1)]

    all_questions = aggregate_block.get("all_questions", {})

    for survey_key, cfg in QUESTION_REGISTRY.items():
        column = cfg["column"]
        dim = cfg["dimension"]

        if column not in company_row:
            continue

        company_value = company_row[column]

        entry = {
            "survey_key": survey_key,
            "column": column,
            "question": cfg["question"],
            "company": company_value,
            "comparison": {},
        }

        # Find sector data
        sector_data = None
        for qblock in all_questions.values():
            if column in qblock:
                sector_data = qblock[column]
                break

        assert sector_data is not None

        if isinstance(company_value, int | float):
            gap = round(company_value - sector_data, 2)
            entry["comparison"] = {
                "type": "numerical",
                "company_value": company_value,
                "value": sector_data,
                "gap": gap,
            }

        else:
            if isinstance(sector_data, float | int):
                entry["comparison"] = {
                    "type": "categorical",
                    "value": sector_data,
                    "sector_distribution": sector_data,
                }
            else:
                # Find the share_pct of the company response in the sector
                match = next(
                    (x for x in sector_data if x["tool"] == company_value), None
                )
                company_pct = match["share_pct"] if match else 0
                entry["comparison"] = {
                    "type": "categorical",
                    "value": company_pct,
                    "sector_distribution": sector_data,
                }

        result[dim].append(entry)

    return result


def get_overall_kpis(aggregate_block: dict):
    results: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))

    for stats in aggregate_block.values():
        for kpi, values in stats["scores"].items():
            for percentile, value in values.items():
                results[kpi][percentile] += value

    total = len(aggregate_block.keys())
    for kpi, values in results.items():
        for percentile, value in values.items():
            results[kpi][percentile] = round(results[kpi][percentile] / total, 2)

    return results


def compare_company_to_aggregate(company_row, aggregate_block):
    return {
        "meta": aggregate_block["meta"],
        "kpis": compare_kpis(company_row, aggregate_block),
        "adoption": compare_adoption(company_row, aggregate_block),
        "market_position": compare_market_tools(company_row, aggregate_block),
        "ordinal_comparison": compare_ordinal(company_row, aggregate_block),
        "responses": compare_responses(company_row, aggregate_block),
        "scoring_logic": get_scoring_logic(),
    }


def get_scoring_logic():
    return {
        "Respuestas Estándar": {
            "Sí": 100,
            "Parcial / En desarrollo": 50,
            "No / Ninguno": 0,
        },
        "Infraestructura TI": {
            "Cloud (Nube)": 100,
            "Híbrida": 70,
            "On-premise (Local)": 30,
        },
        "Inteligencia Artificial": {
            "En producción": 100,
            "En piloto": 75,
            "Explorando": 40,
            "No": 0,
        },
        "Ingresos / Digitalización": {
            "Alto (>60% / 76-100%)": 100,
            "Medio (30-60% / 51-75%)": 75,
            "Bajo (10-30% / 26-50%)": 50,
            "Nulo (<10% / 0-25%)": 25,
        },
    }
