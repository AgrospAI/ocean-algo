import datetime

from src.aggregation.charts import (
    create_age_cloud_chart,
    create_boxplot_chart,
    create_bubble_chart,
    create_digital_traction_chart,
    create_distplot_chart,
    create_erp_crm_distplot_chart,
    create_int_driver_charts,
    create_int_opportunies_charts,
    create_int_priority_chart,
    create_int_radar_chart,
    create_interactive_map,
    create_pie_chart,
    create_riskbar_chart,
    generate_dynamic_conclusion,
)


def generate_interactive_report(df):
    scoring_maps = {
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

    risk_pct = round((len(df[df["KPI_SECURITY"] < 40]) / len(df)) * 100, 1)
    conclusion_data = generate_dynamic_conclusion(df)

    return {
        "date": datetime.date.today().strftime("%d %b %Y"),
        "total_companies": len(df),
        "avg_score": round(df["GLOBAL_SCORE"].mean(), 1),
        "risk_pct": risk_pct,
        "scoring_logic": scoring_maps,
        "chart_radar_url": create_int_radar_chart(df),
        "chart_distplot_url": create_distplot_chart(df),
        "chart_boxplot_url": create_boxplot_chart(df),
        "chart_riskbar_url": create_riskbar_chart(df),
        "chart_priority_url": create_int_priority_chart(df),
        "chart_pie_erp_url": create_pie_chart(df)[0],
        "chart_pie_crm_url": create_pie_chart(df)[1],
        "chart_driver_erp_url": create_int_driver_charts(df)[0],
        "chart_driver_crm_url": create_int_driver_charts(df)[1],
        "chart_driver_cloud_url": create_int_driver_charts(df)[2],
        "chart_erp_distplot_url": create_erp_crm_distplot_chart(df)[0],
        "chart_crm_distplot_url": create_erp_crm_distplot_chart(df)[1],
        "chart_usage_crm_url": create_int_opportunies_charts(df)[0],
        "chart_usage_erp_url": create_int_opportunies_charts(df)[1],
        "chart_region_url": create_interactive_map(df),
        "chart_bubble_url": create_bubble_chart(df),
        "chart_age_cloud_url": create_age_cloud_chart(df),
        "chart_digital_traction_url": create_digital_traction_chart(df),
        "conclusion": conclusion_data,
    }
