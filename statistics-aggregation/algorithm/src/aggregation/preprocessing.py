import re
import unicodedata
from pathlib import Path
from typing import List, Tuple

import pandas as pd


class Preprocessing:
    def __init__(self, config: dict):
        self.config = config

    def _read_parse_csv(self, file_path: Path, sep: str = ";") -> pd.DataFrame:
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

    def _normalize_questions_id(self, text: str) -> str | None:
        """
        Standardizes question string to create a unique key.
        """
        if pd.isna(text) or re.match(r"^\d", str(text)):
            return None

        text = str(text).lower().strip()
        text = re.sub(r"\s*\(.*?\)", "", text)
        text = (
            unicodedata.normalize("NFD", text).encode("ascii", "ignore").decode("ascii")
        )
        text = text.replace("/", " ").replace("-", " ").replace(".", " ")
        text = re.sub(r'"', "", text)
        text = re.sub(r"[^a-z\s]", "", text)
        text = re.sub(r"\s+", "_", text)
        return text[:50]

    def _normalize_free_text(self, text: str, rule_type: str) -> str:
        """
        Basing on question value type access to keywords dictionary and categorize them.
        """
        if not text or str(text).lower() == "nan":
            return "No especificado"

        clean_text = str(text).lower().strip()
        rules = self.config.get(rule_type, {})

        for category, keywords in rules.items():
            for keyword in keywords:
                if keyword in clean_text:
                    return category

        return "Otro"

    def _normalize_response_value(self, row: pd.Series) -> str | int | float:
        """
        Normalize raw CSV answers into clean Data.
        """
        raw_val = str(row["Valor"]).strip()
        field_id = str(row.get("internal_id", ""))

        schema_options = self.config.get("survey_schema", {}).get(field_id, {})
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
                return self.config["cnae_map"].get(raw_val, "Otro")
            return self._normalize_free_text(raw_val, rule_type="sector")

        if field_id in ["erp_in_use", "crm_in_use", "powerbi_usage"]:
            # Check if value is in predefined csv questions options if not use keyword dictionary
            if raw_val in schema_options:
                return raw_val
            return self._normalize_free_text(raw_val, rule_type="software")

        if "channel" in field_id or "comunicacion" in field_id:
            return self._normalize_free_text(raw_val, rule_type="channel")

        if "antivirus" in field_id:
            return self._normalize_free_text(raw_val, rule_type="antivirus")

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
        self, files_path: List[Tuple[str, Path]], csv_separator: str = ";"
    ) -> pd.DataFrame:
        """
        Loops through CSVs, merges with schema, cleans data and unifies client data.
        """
        all_data = []
        print(f"Found {len(files_path)} files")

        for file_name, file_path in files_path:
            try:
                answers_df = self._read_parse_csv(file_path, sep=csv_separator)
                if answers_df.empty or "Campo" not in answers_df.columns:
                    continue

                answers_df["questions"] = answers_df["Campo"].apply(
                    self._normalize_questions_id
                )
                answers_df = answers_df.dropna(subset=["questions"])
                answers_df["internal_id"] = (
                    answers_df["questions"]
                    .map(self.config.get("quest_mapping", {}))
                    .fillna(answers_df["questions"])
                )
                answers_df["normalized_value"] = answers_df.apply(
                    self._normalize_response_value, axis=1
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

    def calculate_maturity_kpis(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates Digital Maturity Scores (0-100)
        """

        # Weights: Processes (40%), Infrastructure (30%), Collaboration (15%), PowerBI (15%)
        df["KPI_OPERATIONS"] = (
            self._get_score("key_processes_digitized_pct", df, "score_percentage")
            * self.config.get("kpi_operation", {}).get("key_processes_digitized_pct", 0)
            + self._get_score("it_infrastructure_type", df, "score_infrastructure")
            * self.config.get("kpi_operation", {}).get("it_infrastructure_type", 0)
            + self._get_score("collaboration_tools_usage", df, "score_binary")
            * self.config.get("kpi_operation", {}).get("collaboration_tools_usage", 0)
            + self._get_score("powerbi_usage", df, "powerbi")
            * self.config.get("kpi_operation", {}).get("powerbi_usage", 0)
        )

        # Weights: Revenue (40%), AI (20%), Web Presence (20%), Marketing (20%)
        df["KPI_BUSINESS"] = (
            self._get_score("digital_revenue", df, "score_revenue")
            * self.config.get("kpi_business", {}).get("digital_revenue", 0)
            + self._get_score("ai_for_automation_usage", df, "score_ai")
            * self.config.get("kpi_business", {}).get("ai_for_automation_usage", 0)
            + self._get_score("active_internet_presence", df, "score_binary")
            * self.config.get("kpi_business", {}).get("active_internet_presence", 0)
            + self._get_score("digital_marketing_use", df, "score_binary")
            * self.config.get("kpi_business", {}).get("digital_marketing_use", 0)
        )

        # 25% each one
        df["KPI_SECURITY"] = (
            self._get_score("two_factor_authentication", df, "score_binary")
            * self.config.get("kpi_security", {}).get("two_factor_authentication", 0)
            + self._get_score("continuity_and_recovery_plans", df, "score_binary")
            * self.config.get("kpi_security", {}).get(
                "continuity_and_recovery_plans", 0
            )
            + self._get_score("phishing_simulations", df, "score_binary")
            * self.config.get("kpi_security", {}).get("phishing_simulations", 0)
            + self._get_score("data_protection_compliance", df, "score_binary")
            * self.config.get("kpi_security", {}).get("data_protection_compliance", 0)
        )

        # Weights: Advanced Skills (25%), Cybersecurity Training (25%), Remote Work Policy (25%), Antivirus Usage (25%)
        df["KPI_CULTURE"] = (
            self._get_score("advanced_digital_skills_pct", df, "score_percentage")
            * self.config.get("kpi_culture", {}).get("advanced_digital_skills_pct", 0)
            + self._get_score("cybersecurity_training", df, "score_binary")
            * self.config.get("kpi_culture", {}).get("cybersecurity_training", 0)
            + self._get_score("remote_work_acceptable_use_policy", df, "score_binary")
            * self.config.get("kpi_culture", {}).get(
                "remote_work_acceptable_use_policy", 0
            )
            + self._get_score("employees_using_antivirus_pct", df, "free_percentage")
            * self.config.get("kpi_culture", {}).get("employees_using_antivirus_pct", 0)
        )

        # Calculate dmi global score
        df["GLOBAL_SCORE"] = (
            df["KPI_OPERATIONS"]
            * self.config.get("global_score", {}).get("KPI_OPERATIONS", 0)
            + df["KPI_SECURITY"]
            * self.config.get("global_score", {}).get("KPI_SECURITY", 0)
            + df["KPI_BUSINESS"]
            * self.config.get("global_score", {}).get("KPI_BUSINESS", 0)
            + df["KPI_CULTURE"]
            * self.config.get("global_score", {}).get("KPI_CULTURE", 0)
        ).round(1)

        df["MATURITY_LABEL"] = df["GLOBAL_SCORE"].apply(self._assign_label)
        return df

    def _get_score(self, col_name, df, map_type):
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
        mapping_dict = self.config.get(map_type, {})
        return df[col_name].map(mapping_dict).fillna(0)

    def _assign_label(self, score):
        """Assign maturity labels based on global score"""
        if score >= 80:
            return "Líder Digital"
        if score >= 60:
            return "Avanzado"
        if score >= 40:
            return "En Desarrollo"
        return "Principiante Digital"
