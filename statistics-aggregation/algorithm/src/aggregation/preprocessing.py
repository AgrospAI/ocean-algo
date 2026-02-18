import re
import unicodedata
from pathlib import Path
from typing import List, Tuple

import pandas as pd

from .config_schema import (
    CNAE_MAP,
    KEYWORD_RULES,
    QUEST_MAPPING,
    SCORING_MAPS,
    SURVEY_SCHEMA,
)


def read_parse_csv(file_path: Path, sep: str = ";") -> pd.DataFrame:
    """
    CSV reader with multiple encoding support.
    """
    encodings = ["utf-8", "latin-1", "iso-8859-1", "cp1252", "utf-16"]

    for encoding in encodings:
        try:
            return pd.read_csv(file_path, encoding=encoding, sep=sep)
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
        # If number of employees is 0, we will consider it as 1 to avoid issues with KPIs calculations
        if "number_of_employees" in field_id and raw_val == "0":
            return 1

        digits = re.sub(r"[^\d\.,]", "", raw_val).replace(",", ".")
        try:
            # If it has decimal, return float else int
            return float(digits) if "." in digits else int(digits)
        except ValueError:
            return 0

    if "cnae" in field_id or "profile" in field_id:
        if raw_val.isdigit() or re.match(r"^-?\d+\.?\d*$", raw_val):
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


def process_surveys(
    files_path: List[Tuple[str, Path]], csv_separator: str = ";"
) -> pd.DataFrame:
    """
    Loops through CSVs, merges with schema, cleans data and unifies client data.
    """
    all_data = []
    # files = glob.glob(os.path.join(data_folder, "*.csv"))
    print(f"Found {len(files_path)} files")

    for file_name, file_path in files_path:
        try:
            answers_df = read_parse_csv(file_path, sep=csv_separator)
            if answers_df.empty or "Campo" not in answers_df.columns:
                continue

            answers_df["questions"] = answers_df["Campo"].apply(normalize_questions_id)
            answers_df = answers_df.dropna(subset=["questions"])
            answers_df["internal_id"] = (
                answers_df["questions"]
                .map(QUEST_MAPPING)
                .fillna(answers_df["questions"])
            )
            answers_df["normalized_value"] = answers_df.apply(
                normalize_response_value, axis=1
            )

            # Create a single row dataframe for this client
            client_row = (
                answers_df[["internal_id", "normalized_value"]]
                .set_index("internal_id")
                .T
            )

            # Add metadata
            client_row.insert(0, "source_file", file_name)

            all_data.append(client_row)

        except Exception as e:
            print(f"Error processing {file_name}: {e}")

    if all_data:
        return pd.concat(all_data, ignore_index=True)
    return pd.DataFrame()


def calculate_maturity_kpis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates Digital Maturity Scores (0-100)
    """

    def get_score(col_name, map_type):
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

    # Weights: Processes (40%), Infrastructure (30%), Collaboration (15%), PowerBI (15%)
    df["KPI_OPERATIONS"] = (
        get_score("key_processes_digitized_pct", "percentage") * 0.4
        + get_score("it_infrastructure_type", "infrastructure") * 0.3
        + get_score("collaboration_tools_usage", "binary") * 0.15
        + get_score("powerbi_usage", "powerbi") * 0.15
    )

    # Weights: Revenue (40%), AI (20%), Web Presence (20%), Marketing (20%)
    df["KPI_BUSINESS"] = (
        get_score("digital_revenue", "revenue") * 0.4
        + get_score("ai_for_automation_usage", "ai") * 0.2
        + get_score("active_internet_presence", "binary") * 0.2
        + get_score("digital_marketing_use", "binary") * 0.2
    )

    # 25% each one
    df["KPI_SECURITY"] = (
        get_score("two_factor_authentication", "binary") * 0.25
        + get_score("continuity_and_recovery_plans", "binary") * 0.25
        + get_score("phishing_simulations", "binary") * 0.25
        + get_score("data_protection_compliance", "binary") * 0.25
    )

    # Weights: Advanced Skills (25%), Cybersecurity Training (25%), Remote Work Policy (25%), Antivirus Usage (25%)
    df["KPI_CULTURE"] = (
        get_score("advanced_digital_skills_pct", "percentage") * 0.25
        + get_score("cybersecurity_training", "binary") * 0.25
        + get_score("remote_work_acceptable_use_policy", "binary") * 0.25
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
