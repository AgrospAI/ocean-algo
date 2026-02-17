import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from numpy import np
from plotly import express as px
from plotly.colors import n_colors

from .config_schema import (
    REGION_COORDINATES,
)

# Set visual style for professional charts
sns.set_theme(style="whitegrid")


def create_int_radar_chart(df):
    categories = [
        "Eficiencia Operativa",
        "Ciberseguridad",
        "Negocio Digital",
        "Cultura Digital",
    ]
    kpi_cols = ["KPI_OPERATIONS", "KPI_SECURITY", "KPI_BUSINESS", "KPI_CULTURE"]

    values_avg = [df[col].mean() for col in kpi_cols]
    values_leaders = [df[col].quantile(0.75) for col in kpi_cols]
    fig = go.Figure()

    fig.add_trace(
        go.Scatterpolar(
            r=values_avg,
            theta=categories,
            fill="toself",
            name="Promedio",
            marker=dict(color="orange"),
        )
    )
    fig.add_trace(
        go.Scatterpolar(
            r=values_leaders,
            theta=categories,
            fill="toself",
            name="Líderes",
            marker=dict(color="purple"),
        )
    )

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
            )
        ),
        showlegend=True,
    )

    return fig.to_html(full_html=False, include_plotlyjs="cdn")


def create_distplot_chart(df):
    all_groups = df.groupby("company_size")["GLOBAL_SCORE"]
    group_labels = ["Micro", "Pequeña", "Mediana", "Grande"]
    existing_labels = []
    hist_data = []

    for label in group_labels:
        if label in all_groups.groups:
            data = all_groups.get_group(label)
            if len(data) > 1:
                hist_data.append(data.to_list())
                existing_labels.append(label)

    colors = n_colors(
        "rgb(5, 200, 200)", "rgb(200, 10, 10)", len(existing_labels), colortype="rgb"
    )

    if not hist_data:
        return (
            "<p>No hay datos suficientes para generar el gráfico de distribución.</p>"
        )

    fig = go.Figure()
    for data, color, label in zip(hist_data, colors, existing_labels):
        fig.add_trace(
            go.Violin(
                x=data,
                name=label,
                line_color=color,
                box_visible=True,
                meanline_visible=True,
            )
        )

    fig.update_traces(orientation="h", side="positive", width=2, points=False)
    fig.update_layout(
        xaxis_showgrid=False,
        xaxis_zeroline=False,
        xaxis_title="Puntuación Global de Madurez Digital",
        yaxis_title="Tamaño de Empresa",
    )

    return fig.to_html(full_html=False, include_plotlyjs="cdn")


def create_boxplot_chart(df):
    order = (
        df.groupby("company_profile_cnae")["GLOBAL_SCORE"]
        .median()
        .sort_values(ascending=False)
        .index
    )
    fig = px.box(
        df,
        x="company_profile_cnae",
        y="GLOBAL_SCORE",
        color="company_profile_cnae",
        category_orders={"company_profile_cnae": order},
        labels={"company_profile_cnae": "Sector", "GLOBAL_SCORE": "Puntuación Global"},
    )
    return fig.to_html(full_html=False, include_plotlyjs="cdn")


def create_riskbar_chart(df):
    risk_cols = [
        "two_factor_authentication",
        "continuity_and_recovery_plans",
        "regular_patching_and_updates",
        "phishing_simulations",
    ]
    risk_labels = ["2FA", "Backups/BCP", "Parcheo", "Phishing Tests"]
    adoption = {}
    for col in risk_cols:
        if col in df.columns:
            adoption[col] = round(
                df[col].astype(str).str.contains(r"sí", case=False).mean() * 100
            )

    fig = go.Figure(
        data=[
            go.Bar(
                y=list(adoption.values()),
                x=risk_labels,
            )
        ],
        layout=dict(
            barcornerradius=10,
        ),
    )

    fig.update_traces(
        marker_color="rgb(158,202,225)",
        marker_line_color="rgb(8,48,107)",
        marker_line_width=1.5,
        opacity=0.6,
    )

    fig.update_layout(
        title="Adopción de Ciberseguridad (%)",
        xaxis=dict(title="Controles de Seguridad"),
        yaxis=dict(title="Porcentaje de Adopción"),
    )

    return fig.to_html(full_html=False, include_plotlyjs="cdn")


def create_int_priority_chart(df):
    if "priority_assessment_area" not in df.columns:
        return "<p>No hay datos suficientes para generar el gráfico de prioridades.</p>"

    priority_counts = df["priority_assessment_area"].value_counts()
    fig = go.Figure(
        data=[
            go.Bar(
                y=priority_counts.index,
                x=priority_counts.values,
                orientation="h",
            )
        ],
    )

    fig.update_traces(
        marker_color="rgb(158,202,225)",
        marker_line_color="rgb(8,48,107)",
        marker_line_width=1.5,
        opacity=0.6,
    )

    fig.update_layout(
        title="Áreas de Evaluación Prioritarias",
        xaxis=dict(title="Número de Empresas"),
        yaxis=dict(title="Área de Evaluación"),
    )

    return fig.to_html(full_html=False, include_plotlyjs="cdn")


def create_pie_chart(df):
    clean_series = df["erp_in_use"].fillna("No especificado")
    counts = clean_series.value_counts().head(6).sort_values()

    fig_erp = go.Figure(
        data=[
            go.Bar(
                x=counts.values,
                y=counts.index,
                orientation="h",
            )
        ]
    )

    fig_erp.update_traces(marker=dict(color=sns.color_palette("pastel").as_hex()))

    fig_erp.update_layout(
        title="Cuota de Mercado ERP",
        xaxis=dict(title="Número de Empresas"),
        yaxis=dict(title="Herramienta"),
    )

    clean_series_crm = df["crm_in_use"].fillna("No especificado")
    counts_crm = clean_series_crm.value_counts().head(6).sort_values()
    fig_crm = go.Figure(
        data=[
            go.Bar(
                x=counts_crm.values,
                y=counts_crm.index,
                orientation="h",
            )
        ]
    )
    fig_crm.update_traces(marker=dict(color=sns.color_palette("pastel").as_hex()))
    fig_crm.update_layout(
        title="Cuota de Mercado CRM",
        xaxis=dict(title="Número de Empresas"),
        yaxis=dict(title="Herramienta"),
    )

    return fig_erp.to_html(full_html=False, include_plotlyjs="cdn"), fig_crm.to_html(
        full_html=False, include_plotlyjs="cdn"
    )


def create_int_driver_charts(df):
    """
    Generates charts showing what drives high maturity (Correlation Analysis).
    """
    # 1. CRM Impact
    df["has_crm"] = df["crm_in_use"].apply(
        lambda x: "Sin CRM" if str(x).lower() in ["ninguno"] else "Con CRM"
    )

    fig_crm = go.Figure()
    for label in df["has_crm"].unique():
        fig_crm.add_trace(
            go.Box(y=df[df["has_crm"] == label]["GLOBAL_SCORE"], name=label)
        )

    fig_crm.update_layout(
        title="Impacto del CRM en la Madurez Global",
        yaxis=dict(title="Puntuación Global"),
        xaxis=dict(title="Uso de CRM"),
    )

    # 2. Infrastructure Impact
    order = ["On-premise", "Híbrida", "Cloud"]
    existing_order = [x for x in order if x in df["it_infrastructure_type"].unique()]

    fig_cloud = go.Figure()
    for label in existing_order:
        fig_cloud.add_trace(
            go.Box(
                y=df[df["it_infrastructure_type"] == label]["GLOBAL_SCORE"], name=label
            )
        )

    fig_cloud.update_layout(
        title="Impacto de la Nube en la Madurez Global",
        yaxis=dict(title="Puntuación Global"),
        xaxis=dict(title="Tipo de Infraestructura TI"),
    )

    # 3. ERP Impact
    df["has_erp"] = df["erp_in_use"].apply(
        lambda x: "Sin ERP" if str(x).lower() in ["ninguno"] else "Con ERP"
    )

    fig_erp = go.Figure()
    for label in df["has_erp"].unique():
        fig_erp.add_trace(
            go.Box(y=df[df["has_erp"] == label]["GLOBAL_SCORE"], name=label)
        )

    fig_erp.update_layout(
        title="Impacto del ERP en la Madurez Global",
        yaxis=dict(title="Puntuación Global"),
        xaxis=dict(title="Uso de ERP"),
    )

    return (
        fig_erp.to_html(full_html=False, include_plotlyjs="cdn"),
        fig_crm.to_html(full_html=False, include_plotlyjs="cdn"),
        fig_cloud.to_html(full_html=False, include_plotlyjs="cdn"),
    )


def create_erp_crm_distplot_chart(df):
    df = df.copy()
    df["has_erp"] = df["erp_in_use"].apply(
        lambda x: "Sin ERP" if str(x).lower() in ["ninguno"] else "Con ERP"
    )
    grouped = df.groupby(["company_size", "has_erp"]).size().reset_index(name="count")
    grouped["percentage"] = grouped.groupby("company_size")["count"].transform(
        lambda x: x / x.sum() * 100
    )

    size_order = ["Micro", "Pequeña", "Mediana", "Grande"]

    fig_erp = px.bar(
        grouped,
        x="percentage",
        y="company_size",
        color="has_erp",
        orientation="h",
        text_auto=".1f",
        category_orders={"company_size": size_order},
        title="Distribución de ERP: Adopción por Tamaño de Empresa",
        labels={
            "percentage": "Porcentaje de Adopción (%)",
            "company_size": "Tamaño de empresa",
            "has_erp": "ERP",
        },
        color_discrete_map={"Con ERP": "#8DE5A1", "Sin ERP": "#CFCFCF"},
    )
    fig_erp.update_layout(legend_title_text="")

    df["has_crm"] = df["crm_in_use"].apply(
        lambda x: "Sin CRM" if str(x).lower() in ["ninguno"] else "Con CRM"
    )
    grouped_crm = (
        df.groupby(["company_size", "has_crm"]).size().reset_index(name="count")
    )
    grouped_crm["percentage"] = grouped_crm.groupby("company_size")["count"].transform(
        lambda x: x / x.sum() * 100
    )
    fig_crm = px.bar(
        grouped_crm,
        x="percentage",
        y="company_size",
        color="has_crm",
        orientation="h",
        text_auto=".1f",
        category_orders={"company_size": size_order},
        title="Distribución de CRM: Adopción por Tamaño de Empresa",
        labels={
            "percentage": "Porcentaje de Adopción (%)",
            "company_size": "Tamaño de empresa",
            "has_crm": "CRM",
        },
        color_discrete_map={"Con CRM": "#8DE5A1", "Sin CRM": "#CFCFCF"},
    )
    fig_crm.update_layout(legend_title_text="")

    return fig_erp.to_html(full_html=False, include_plotlyjs="cdn"), fig_crm.to_html(
        full_html=False, include_plotlyjs="cdn"
    )


def create_int_opportunies_charts(df):
    no_crm_mask = df["crm_in_use"].astype(str).str.contains("ninguno", case=False)
    no_crm_count = no_crm_mask.sum()
    total_count = len(df)
    has_crm_count = total_count - no_crm_count

    sizes = [no_crm_count, has_crm_count]
    labels = [f"Sin CRM ({no_crm_count})", f"Con CRM ({has_crm_count})"]
    colors = ["#CFCFCF", "#8DE5A1"]

    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels, values=sizes, marker=dict(colors=colors), pull=[0, 0.1]
            )
        ]
    )

    fig.update_layout(title="Mercado CRM")

    no_erp_mask = df["erp_in_use"].astype(str).str.contains("ninguno", case=False)
    no_erp_count = no_erp_mask.sum()
    has_erp_count = total_count - no_erp_count

    sizes_erp = [no_erp_count, has_erp_count]
    labels_erp = [f"Sin ERP ({no_erp_count})", f"Con ERP ({has_erp_count})"]
    colors_erp = ["#CFCFCF", "#8DE5A1"]
    fig_erp = go.Figure(
        data=[
            go.Pie(
                labels=labels_erp,
                values=sizes_erp,
                marker=dict(colors=colors_erp),
                pull=[0, 0.1],
            )
        ]
    )

    fig_erp.update_layout(title="Mercado ERP")

    return fig.to_html(full_html=False, include_plotlyjs="cdn"), fig_erp.to_html(
        full_html=False, include_plotlyjs="cdn"
    )


def create_interactive_map(df):
    """
    Generates a Bubble Map of companies.
    """
    df = df[["company_profile_cnae", "province", "GLOBAL_SCORE"]].copy()
    df = df.groupby("province")["GLOBAL_SCORE"].mean().round(1).reset_index()

    def get_lat(row):
        return REGION_COORDINATES.get(
            row.get("province", "Unknown"), REGION_COORDINATES["Unknown"]
        )["lat"]

    def get_lon(row):
        return REGION_COORDINATES.get(
            row.get("province", "Unknown"), REGION_COORDINATES["Unknown"]
        )["lon"]

    df["lat_base"] = df.apply(get_lat, axis=1)
    df["lon_base"] = df.apply(get_lon, axis=1)

    fig = px.scatter_map(
        df,
        lat="lat_base",
        lon="lon_base",
        color="GLOBAL_SCORE",
        size=df["GLOBAL_SCORE"] * 0.3,
        color_continuous_scale=px.colors.sequential.Plasma,
        size_max=30,
        zoom=5,
        hover_name="province",
        hover_data={
            "lat_base": False,
            "lon_base": False,
            "GLOBAL_SCORE": True,
            "province": True,
        },
        map_style="streets",
        title="Mapa de Madurez Digital (Geolocalizado)",
        labels={"GLOBAL_SCORE": "Madurez Digital", "province": "Provincia"},
    )

    fig.update_layout(height=700, margin={"r": 0, "t": 40, "l": 0, "b": 0})

    return fig.to_html(full_html=False, include_plotlyjs="cdn")


def create_bubble_chart(df):
    df = df.copy()
    df["number_of_employees"] = pd.to_numeric(
        df["number_of_employees"], errors="coerce"
    )
    df = df[df["number_of_employees"] > 0]

    def estimate_revenue(val):
        s = str(val)
        if "<1M" in s:
            return 500_000
        if "1-5M" in s:
            return 3_000_000
        if "5-20M" in s:
            return 12_500_000
        if ">20M" in s:
            return 50_000_000
        return np.nan

    df["est_revenue"] = df["annual_revenue"].apply(estimate_revenue)
    df["revenue_per_employee"] = df["est_revenue"] / df["number_of_employees"]
    plot_df = df.dropna(
        subset=["revenue_per_employee", "GLOBAL_SCORE", "number_of_employees"]
    )

    if plot_df.empty:
        return "<div>No hay datos suficientes para calcular la eficiencia.</div>"

    # Create descriptive company size labels with employee ranges
    size_mapping = {
        "Micro": "Micro (≤10 empleados)",
        "Pequeña": "Pequeña (11-50 empleados)",
        "Mediana": "Mediana (51-250 empleados)",
        "Grande": "Grande (>250 empleados)",
    }
    plot_df = plot_df.copy()
    plot_df["company_size_desc"] = plot_df["company_size"].map(size_mapping)

    fig = px.scatter(
        plot_df,
        x="GLOBAL_SCORE",
        y="revenue_per_employee",
        size="number_of_employees",
        color="company_size_desc",
        hover_name="company_profile_cnae",
        hover_data={
            "company_size": False,
            "company_size_desc": False,
            "number_of_employees": ":.0f",
            "annual_revenue": True,
            "revenue_per_employee": ":.2s",
            "GLOBAL_SCORE": True,
        },
        labels={
            "GLOBAL_SCORE": "Índice de Madurez Digital (0-100)",
            "number_of_employees": "Número de Empleados",
            "annual_revenue": "Rango de Facturación Anual",
            "revenue_per_employee": "Ingresos Estimados / Empleado (€)",
            "company_size_desc": "Tamaño de Empresa",
        },
        color_discrete_map={
            "Micro (≤10 empleados)": "#2ecc71",
            "Pequeña (11-50 empleados)": "#3498db",
            "Mediana (51-250 empleados)": "#9b59b6",
            "Grande (>250 empleados)": "#e74c3c",
        },
    )

    fig.update_layout(
        height=600, yaxis=dict(tickprefix="€"), legend=dict(orientation="h", y=-0.20)
    )

    return fig.to_html(full_html=False, include_plotlyjs="cdn")


def create_age_cloud_chart(df):
    fig = go.Figure(
        data=go.Heatmap(
            z=pd.crosstab(df["average_employee_age"], df["it_infrastructure_type"]),
            x=["On-premise", "Híbrida", "Cloud"],
            y=["<30", "30-40", "41-50", ">50"],
            colorscale="Reds",
        )
    )

    fig.update_layout(
        xaxis_title="Tipo de Infraestructura TI", yaxis_title="Rango de Edad"
    )

    return fig.to_html(full_html=False, include_plotlyjs="cdn")


def create_digital_traction_chart(df):
    df = df.copy()

    def clean_digital_rev(val):
        s = str(val)
        if "<10%" in s:
            return 5
        if "10-30%" in s:
            return 20
        if "30-60%" in s:
            return 45
        if ">60%" in s:
            return 80
        if "Nulo" in s or "0" in s:
            return 0
        return 0

    df["est_digital_revenue_pct"] = df["digital_revenue"].apply(clean_digital_rev)
    df["marketing_status"] = df["digital_marketing_use"].fillna("No")

    fig = px.box(
        df,
        x="marketing_status",
        y="est_digital_revenue_pct",
        color="marketing_status",
        points="all",
        hover_name="company_profile_cnae",
        hover_data={
            "annual_revenue": True,
            "digital_revenue": True,
            "est_digital_revenue_pct": False,
            "marketing_status": False,
        },
        labels={
            "marketing_status": "¿Invierte en Marketing Digital?",
            "est_digital_revenue_pct": "% Estimado de Facturación Online",
        },
        color_discrete_map={"Sí": "#2ecc71", "No": "#95a5a6", "Parcial": "#3498db"},
    )

    fig.update_layout(
        height=400, yaxis=dict(ticksuffix="%", range=[-5, 100]), showlegend=False
    )

    return fig.to_html(full_html=False, include_plotlyjs="cdn")


def generate_dynamic_conclusion(df):
    avg_score = df["GLOBAL_SCORE"].mean().round(1)
    dims = {
        "Operaciones": df["KPI_OPERATIONS"].mean().round(1),
        "Ciberseguridad": df["KPI_SECURITY"].mean().round(1),
        "Negocio Digital": df["KPI_BUSINESS"].mean().round(1),
        "Cultura Digital": df["KPI_CULTURE"].mean().round(1),
    }

    strong_dim = max(dims, key=dims.get)
    weakest_dim = min(dims, key=dims.get)

    if "priority_assessment_area" in df.columns:
        top_priority = df["priority_assessment_area"].mode()[0]
    else:
        top_priority = "No especificado"

    if avg_score < 40:
        verdict_title = "Mercado en Etapa Inicial"
        verdict_text = f"El índice de madurez promedio ({avg_score}) indica que el sector se encuentra en una fase temprana de digitalización. La tecnología se usa de forma reactiva, no estratégica."
        color_class = "#e74c3c"  # Red
    elif avg_score < 70:
        verdict_title = "Mercado en Desarrollo"
        verdict_text = f"Con un índice de {avg_score}, el mercado muestra avances sólidos, aunque dispares. Las empresas han adoptado herramientas básicas pero faltan procesos integrados."
        color_class = "#f39c12"  # Orange
    else:
        verdict_title = "Mercado Maduro"
        verdict_text = f"El sector demuestra una alta competencia digital ({avg_score}). El desafío ya no es la adopción, sino la innovación y el uso de IA."
        color_class = "#2ecc71"  # Green

    gap_text = f"El análisis revela que <strong>{weakest_dim}</strong> es el área crítica de mejora (puntuación: {dims[weakest_dim]}). Mientras que {strong_dim} actúa como motor, el descuido en {weakest_dim} está frenando el crecimiento global."

    if top_priority in weakest_dim:
        strategy_text = f"Es positivo notar que las empresas son conscientes de su debilidad: la mayoría ha marcado <strong>{top_priority}</strong> como su prioridad, alineándose con los datos."
    else:
        strategy_text = f"Existe una <strong>disonancia estratégica</strong>: Aunque la mayor debilidad objetiva es {weakest_dim}, las empresas están priorizando invertir en <strong>{top_priority}</strong>. Se recomienda reevaluar este enfoque para cerrar brechas estructurales antes de buscar crecimiento."

    html_content = f"""
    <div style="background: white; padding: 25px; border-left: 6px solid {color_class}; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.05);">
        <h3 style="color: {color_class}; margin-top: 0;">{verdict_title} (Promedio: {avg_score}/100)</h3>
        <p style="margin-bottom: 15px;">{verdict_text}</p>
        
        <h4 style="color: #0062a4; margin-bottom: 5px;">El factor limitante: {weakest_dim}</h4>
        <p style="margin-bottom: 15px;">{gap_text}</p>
        
        <h4 style="color: #0062a4; margin-bottom: 5px;">Recomendación Estratégica</h4>
        <p style="margin-bottom: 0;">{strategy_text}</p>
    </div>
    """

    return {
        "avg_score": avg_score,
        "verdict_title": verdict_title,
        "verdict_text": verdict_text,
        "color_class": color_class,
        "weakest_dim": weakest_dim,
        "gap_text": gap_text,
        "strategy_text": strategy_text,
        "html_content": html_content,
    }
