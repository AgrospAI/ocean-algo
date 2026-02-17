from collections import defaultdict

import numpy as np
import pandas as pd

from .config_schema import POSTAL_CODE, SECTION_MAPPING, SURVEY_SCHEMA


def generate_benchmark_reference(df: pd.DataFrame) -> dict:
    """
    Takes the master dataframe with dmi calculated and aggregates it into
    a json dictionary for benchmarking.
    """

    # Establish company size categories
    conditions = [
        (df["number_of_employees"] <= 10),
        (df["number_of_employees"] > 10) & (df["number_of_employees"] <= 50),
        (df["number_of_employees"] > 50) & (df["number_of_employees"] <= 250),
        (df["number_of_employees"] > 250),
    ]
    choices = ["Micro", "Pequeña", "Mediana", "Grande"]
    df["company_size"] = np.select(conditions, choices, default="Desconocido")

    df["province"] = (
        df["company_postcode"]
        .astype(str)
        .str[:2]
        .map(POSTAL_CODE)
        .fillna("Desconocido")
    )

    reference_data = {}

    # Group by sector + size ("Industrial" + "Small")
    groups_combinations = [
        df.groupby(["company_profile_cnae", "company_size"]),
        df.groupby(["company_profile_cnae", "province"]),
        df.groupby(["company_size", "province"]),
    ]

    print(f"Generating benchmarks for {len(groups_combinations)}...")

    for grouped in groups_combinations:
        reference_data.update(calculate_reference_by_group(grouped))

    return reference_data


def calculate_reference_by_group(grouped) -> dict:
    """
    Generic function to calculate reference data by any grouping columns.
    """
    score_cols = [
        "KPI_OPERATIONS",
        "KPI_SECURITY",
        "KPI_BUSINESS",
        "KPI_CULTURE",
        "GLOBAL_SCORE",
    ]

    # We check the % of companies that have these implemented (yes/no)
    adoption_cols = [
        "two_factor_authentication",
        "continuity_and_recovery_plans",
        "microsoft_365_usage",
        "remote_work_acceptable_use_policy",
        "phishing_simulations",
        "active_internet_presence",
        "data_protection_compliance",
        "regular_patching_and_updates",
        "incident_response_plan",
        "digital_marketing_use",
        "accessible_digital_sales_channels",
        "continuous_digital_training",
        "ai_for_automation_usage",
    ]

    # We check the most popular tools in each sector(top 3)
    market_cols = [
        "erp_in_use",
        "crm_in_use",
        "it_infrastructure_type",
        "antivirus_used",
        "powerbi_usage",
        "priority_assessment_area",
        "average_employee_age",
        "it_outsourcing_level",
        "digital_revenue",
    ]

    reference_data = {}
    outsourcing_map = {"Bajo": 1, "Medio": 2, "Alto": 3}

    for (x, y), group in grouped:
        # Unique ID (eg. "Industrial_Small")
        combination_id = f"{x}_{y}"
        column_names = grouped.grouper.names

        # Minimum 3 samples to consider valid benchmark
        if len(group) < 3:
            continue

        stats = {
            "meta": {column_names[0]: x, column_names[1]: y, "sample_size": len(group)},
            "scores": {},
            "adoption_rates": {},
            "market_leaders": {},
            "averages": {},
            "all_questions": defaultdict(dict),
        }

        for col in group.columns:
            if col in score_cols:
                score_stats = calculate_score_stats(group[col])
                stats["scores"][col] = score_stats

            elif col in adoption_cols:
                adoptation_rate_stats = calculate_adoption_rate(
                    group[col], keyword="Sí|En producción"
                )
                stats["adoption_rates"][col] = adoptation_rate_stats
                stats["all_questions"][SECTION_MAPPING[col]][col] = (
                    adoptation_rate_stats
                )

            elif col in market_cols:
                stats["market_leaders"][col] = format_top_items(group[col])
                stats["all_questions"][SECTION_MAPPING[col]][col] = format_top_items(
                    group[col]
                )

            elif SURVEY_SCHEMA.get(col) == ["NUMERIC"]:
                avg_val = group[col].mean()
                stats["averages"][col] = float(round(avg_val, 2))
                stats["all_questions"][SECTION_MAPPING[col]][col] = float(
                    round(avg_val, 2)
                )

            elif SURVEY_SCHEMA.get(col) == ["TEXT"]:
                stats["all_questions"][SECTION_MAPPING[col]][col] = format_top_items(
                    group[col]
                )

            elif SURVEY_SCHEMA.get(col):
                stats["all_questions"][SECTION_MAPPING[col]][col] = format_top_items(
                    group[col]
                )

        if "it_outsourcing_level" in group.columns:
            numeric_vals = group["it_outsourcing_level"].map(outsourcing_map).dropna()
            if not numeric_vals.empty:
                avg_val = numeric_vals.mean()
                # Convert back to text for display (1=Bajo, 2=Medio, 3=Alto)
                label = (
                    "Bajo" if avg_val < 1.5 else "Alto" if avg_val > 2.5 else "Medio"
                )
                stats["averages"]["it_outsourcing_level"] = {
                    "avg_score": float(round(avg_val, 2)),
                    "label": label,
                }

        reference_data[combination_id] = stats
    return reference_data


def calculate_score_stats(series):
    """Calculate p25, median, p75 for a numeric series."""
    return {
        "p25": float(round(series.quantile(0.25), 1)),
        "median": float(round(series.median(), 1)),
        "p75": float(round(series.quantile(0.75), 1)),
    }


def calculate_adoption_rate(series, keyword):
    """Calculate percentage of rows containing keyword."""
    is_match = series.astype(str).str.contains(keyword, case=False, regex=True)
    return float(round(is_match.mean() * 100, 1))


def format_top_items(series, n=3):
    """Format top N items by frequency as list of dicts."""
    counts = series.value_counts(normalize=True).head(n)
    return [
        {"tool": name, "share_pct": float(round(share * 100, 1))}
        for name, share in counts.items()
    ]
