import os
import shutil
import tempfile
from logging import getLogger
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates

# ReportLab Imports
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, 
    Table, TableStyle, PageBreak, KeepTogether
)
from reportlab.lib import colors
from reportlab.lib.units import cm

# --- OCEAN RUNNER IMPORTS ---
from ocean_runner import Algorithm, Config
from .data import InputParameters

logger = getLogger(__name__)

# 1. SETUP OCEAN RUNNER
algorithm = Algorithm(config=Config(custom_input=InputParameters))


# 2. BUSINESS LOGIC CLASS
class PDFReportGenerator:
    """
    Encapsulates the logic to generate the PDF.
    Decoupled from Ocean Protocol specifics (receives 'params' directly).
    """
    def __init__(self, params: InputParameters):
        self.params = params  # Replaces self._job_details.input_parameters

    def generate(self, df: pd.DataFrame, csv_name: str, logo_path: Optional[Path], output_dir: Path) -> Path:
        """
        Main orchestrator. Returns path to the generated PDF.
        """
        # 1. Pre-processing
        datetime_col = self._find_datetime_column(df)
        df[datetime_col] = pd.to_datetime(df[datetime_col])
        df = df.sort_values(datetime_col).reset_index(drop=True)

        numeric_cols = self._find_numeric_columns(df, datetime_col)
        if not numeric_cols:
            raise ValueError("No numeric columns found in CSV.")

        # 2. Metadata & Statistics
        date_min = df[datetime_col].min().strftime("%Y-%m-%d %H:%M")
        date_max = df[datetime_col].max().strftime("%Y-%m-%d %H:%M")

        df["__time_diff__"] = df[datetime_col].diff().dt.total_seconds().fillna(0)
        median_interval = df["__time_diff__"].median()
        missing_gaps = df[df["__time_diff__"] > 1.5 * median_interval][[datetime_col, "__time_diff__"]]

        summary_stats = df[numeric_cols].describe()

        # 3. Generate Plots (Dynamic window calculation)
        win = max(5, min(int(len(df) * 0.1), 50))
        
        # 
        hist_paths = self._generate_histograms(df, numeric_cols, output_dir)
        ts_paths = self._generate_time_series_plots(df, numeric_cols, datetime_col, win, output_dir)
        scatter_matrix_path, corr_matrix_path = self._generate_correlation_plots(df, numeric_cols, output_dir)
        diurnal_paths = self._generate_diurnal_plots(df, numeric_cols, datetime_col, output_dir)
        anomalies_info = self._detect_anomalies(df, numeric_cols, datetime_col)
        
        # 4. Build PDF
        report_filename = "final_report.pdf"
        pdf_path = output_dir / report_filename
        
        self._build_pdf(
            csv_name=csv_name,
            logo_path=logo_path,
            report_pdf=pdf_path,
            date_min=date_min,
            date_max=date_max,
            numeric_cols=numeric_cols,
            median_interval=median_interval,
            missing_gaps=missing_gaps,
            summary_stats=summary_stats,
            hist_paths=hist_paths,
            ts_paths=ts_paths,
            win=win,
            scatter_matrix_path=scatter_matrix_path,
            corr_matrix_path=corr_matrix_path,
            diurnal_paths=diurnal_paths,
            anomalies_info=anomalies_info,
            df=df,
            datetime_col=datetime_col,
            output_dir=output_dir,
        )

        return pdf_path

    # ────────────── Helper Methods (Data & Plotting) ──────────────

    def _find_datetime_column(self, df: pd.DataFrame) -> str:
        for col in df.columns:
            try:
                pd.to_datetime(df[col])
                return col
            except Exception:
                continue
        raise ValueError("No column could be parsed as datetime.")

    def _find_numeric_columns(self, df: pd.DataFrame, datetime_col: str) -> List[str]:
        numerics: List[str] = []
        for col in df.columns:
            if col == datetime_col:
                continue
            try:
                df[col] = pd.to_numeric(df[col])
                numerics.append(col)
            except Exception:
                continue
        return numerics

    def _generate_histograms(self, df: pd.DataFrame, numeric_cols: List[str], output_dir: Path) -> Dict[str, Path]:
        hist_paths: Dict[str, Path] = {}
        for col in numeric_cols:
            try:
                fig, ax = plt.subplots(figsize=(4, 2.8))
                ax.hist(df[col].dropna(), bins=20, color="tab:blue", alpha=0.7)
                ax.set_title(f"Histogram of {col}")
                ax.set_xlabel(col)
                ax.set_ylabel("Count")
                plt.tight_layout()
                p = output_dir / f"hist_{col}.png"
                fig.savefig(p, dpi=120)
                plt.close(fig)
                hist_paths[col] = p
            except Exception as e:
                logger.warning(f"Could not plot histogram for '{col}': {e}")
        return hist_paths

    def _generate_time_series_plots(self, df: pd.DataFrame, numeric_cols: List[str], datetime_col: str, win: int, output_dir: Path) -> Dict[str, Path]:
        ts_paths: Dict[str, Path] = {}
        for col in numeric_cols:
            try:
                fig, ax = plt.subplots(figsize=(8, 3.5))
                ax.plot(df[datetime_col], df[col], label=f"{col} (raw)", color="tab:red", alpha=0.6)
                ax.plot(df[datetime_col], df[col].rolling(window=win, min_periods=1).mean(), label=f"{col} {win}-pt MA", color="tab:orange", linestyle="--")
                ax.legend(fontsize=8)
                plt.tight_layout()
                
                locator = mdates.AutoDateLocator(minticks=4, maxticks=8)
                formatter = mdates.AutoDateFormatter(locator)
                ax.xaxis.set_major_locator(locator)
                ax.xaxis.set_major_formatter(formatter)
                fig.autofmt_xdate(rotation=30, ha="right")
                
                p = output_dir / f"ts_{col}.png"
                fig.savefig(p, dpi=120)
                plt.close(fig)
                ts_paths[col] = p
            except Exception as e:
                logger.warning(f"Could not plot time series for '{col}': {e}")
        return ts_paths

    def _generate_correlation_plots(self, df: pd.DataFrame, numeric_cols: List[str], output_dir: Path) -> Tuple[Optional[Path], Optional[Path]]:
        if len(numeric_cols) < 2: return None, None
        scatter_path, corr_path = None, None
        try:
            sns.set(style="whitegrid")
            g = sns.pairplot(df[numeric_cols].dropna(), kind="scatter", plot_kws={"s": 10, "alpha": 0.6})
            scatter_path = output_dir / "scatter_matrix.png"
            g.figure.savefig(scatter_path, dpi=120)
            plt.close(g.figure)
        except Exception: pass
        
        try:
            corr = df[numeric_cols].corr()
            fig, ax = plt.subplots(figsize=(4 + 0.5 * len(numeric_cols), 4))
            sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", cbar_kws={"shrink": 0.5}, ax=ax, linewidths=0.3)
            ax.set_title("Correlation Matrix")
            plt.tight_layout()
            corr_path = output_dir / "corr_matrix.png"
            fig.savefig(corr_path, dpi=120)
            plt.close(fig)
        except Exception: pass
        return scatter_path, corr_path

    def _generate_diurnal_plots(self, df: pd.DataFrame, numeric_cols: List[str], datetime_col: str, output_dir: Path) -> Dict[str, Path]:
        df["__hour__"] = df[datetime_col].dt.hour
        diurnal_paths = {}
        # UPDATED: Use self.params
        ticks = range(self.params.diurnal_xticks.min, self.params.diurnal_xticks.max, self.params.diurnal_xticks.step)
        
        for col in numeric_cols:
            try:
                hourly_avg = df.groupby("__hour__")[col].mean().reset_index()
                fig, ax = plt.subplots(figsize=(6, 2.8))
                ax.plot(hourly_avg["__hour__"], hourly_avg[col], marker="o", color="tab:green")
                ax.set_xticks(ticks)
                ax.set_xlabel("Hour of Day")
                ax.set_ylabel(f"Avg {col}")
                ax.set_title(f"Diurnal Pattern of {col}")
                plt.tight_layout()
                p = output_dir / f"diurnal_{col}.png"
                fig.savefig(p, dpi=120)
                plt.close(fig)
                diurnal_paths[col] = p
            except Exception: pass
        return diurnal_paths

    def _detect_anomalies(self, df: pd.DataFrame, numeric_cols: List[str], datetime_col: str):
        anomalies_info = {}
        for col in numeric_cols:
            col_std = df[col].std()
            df[f"__delta_{col}__"] = df[col].diff().abs().fillna(0)
            # UPDATED: Use self.params
            threshold = self.params.anomaly_multiplier * col_std
            jumps = df[df[f"__delta_{col}__"] > threshold][[datetime_col, col, f"__delta_{col}__"]]
            anomalies_info[col] = jumps.to_dict(orient="records")
        return anomalies_info

    # ────────────── PDF Building Methods ──────────────

    def _get_report_styles(self):
        styles = getSampleStyleSheet()
        styles["Title"].fontName = "Helvetica-Bold"
        styles["BodyText"].fontName = "Helvetica"
        styles.add(ParagraphStyle(name="SectionHeader", fontSize=14, fontName="Helvetica-Bold", textColor=colors.HexColor("#457B9D"), spaceAfter=6))
        styles.add(ParagraphStyle(name="SubHeader", fontSize=12, fontName="Helvetica-Bold", textColor=colors.HexColor("#1D3557"), spaceAfter=4))
        styles.add(ParagraphStyle(name="BodyTextSmall", fontSize=9, fontName="Helvetica", leading=11))
        return styles

    def _image_with_caption(self, img_path, caption_text, width_cm, height_cm, styles):
        img = Image(str(img_path), width=width_cm * cm, height=height_cm * cm)
        caption = Paragraph(caption_text, styles["BodyTextSmall"])
        tbl = Table([[img], [caption]], colWidths=[width_cm * cm], hAlign="CENTER")
        tbl.setStyle(TableStyle([("ALIGN", (0, 0), (-1, -1), "CENTER"), ("VALIGN", (0, 0), (-1, -1), "TOP"), ("BOTTOMPADDING", (0, 0), (0, 0), 4)]))
        return tbl

    def _append_cover_page(self, story, logo_path, csv_name, date_min, date_max, numeric_cols, styles):
        if logo_path:
            story.append(Image(str(logo_path), width=6 * cm, height=6 * cm))
            story.append(Spacer(1, 1.5 * cm))
        story.append(Paragraph(f"<b><font size=24 color='#1D3557'>Data Report: {csv_name}</font></b>", styles["Title"]))
        story.append(Spacer(1, 0.7 * cm))
        story.append(Paragraph(f"<font size=12 color='#457B9D'>Date Range: {date_min} - {date_max}</font>", styles["BodyText"]))
        story.append(Spacer(1, 0.3 * cm))
        story.append(Paragraph(f"<font size=12 color='#457B9D'>Numeric Columns ({len(numeric_cols)}): {', '.join(numeric_cols)}</font>", styles["BodyText"]))
        story.append(Spacer(1, 0.5 * cm))
        story.append(Paragraph(f"<font size=10 color='#A8A8A8'>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}</font>", styles["BodyTextSmall"]))
        story.append(Spacer(1, 2 * cm))
        story.append(Paragraph("<font size=9 color='#A8A8A8'>Confidential - Internal Use Only</font>", styles["BodyTextSmall"]))
        story.append(PageBreak())

    def _header_footer_callback(self, canvas, doc, csv_name):
        canvas.saveState()
        if doc.page > 1:
            header_text = f"Data Report: {csv_name}"
            canvas.setFont("Helvetica-Bold", 9)
            canvas.setFillColor(colors.HexColor("#457B9D"))
            canvas.drawString(1.5 * cm, A4[1] - 1.2 * cm, header_text)
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(colors.HexColor("#A8A8A8"))
        page_num = f"Page {doc.page}"
        canvas.drawCentredString(A4[0] / 2, 1 * cm, page_num)
        canvas.drawRightString(A4[0] - 1.5 * cm, 1 * cm, "Confidential - Internal Use Only")
        canvas.restoreState()

    def _build_section_data_quality(self, story, median_interval, missing_gaps, datetime_col, styles):
        story.append(Paragraph("<b><font size=16 color='#457B9D'>Section 1: Data Quality</font></b>", styles["SectionHeader"]))
        story.append(Spacer(1, 0.5 * cm))
        story.append(Paragraph(f"Median sampling interval: {median_interval:.1f} seconds", styles["BodyTextSmall"]))
        story.append(Spacer(1, 0.2 * cm))
        if not missing_gaps.empty:
            story.append(Paragraph("<b><font size=13 color='#1D3557'>Large Sampling Gaps (time -> gap_seconds):</font></b>", styles["SubHeader"]))
            for _, row in missing_gaps.iterrows():
                ts_str = row[datetime_col].strftime("%Y-%m-%d %H:%M:%S")
                gap = row["__time_diff__"]
                story.append(Paragraph(f"{ts_str} -> gap = {gap:.1f} s", styles["BodyTextSmall"]))
        else:
            story.append(Paragraph("No large sampling gaps detected.", styles["BodyTextSmall"]))
        story.append(PageBreak())

    def _build_section_summary_stats(self, story, numeric_cols, summary_stats, styles):
        story.append(Paragraph("<b><font size=16 color='#457B9D'>Section 2: Summary Statistics</font></b>", styles["SectionHeader"]))
        story.append(Spacer(1, 0.5 * cm))
        metrics = ["min", "25%", "50%", "mean", "75%", "max", "std"]
        table_data = [["Metric"] + numeric_cols]
        for m in metrics:
            row = [m.capitalize()] + [f"{summary_stats.at[m, col]:.2f}" for col in numeric_cols]
            table_data.append(row)
        col_width = (A4[0] - 3 * cm) / (len(numeric_cols) + 1)
        tbl = Table(table_data, colWidths=[col_width] * (len(numeric_cols) + 1), hAlign="CENTER")
        tbl.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F1FAEE")), ("GRID", (0, 0), (-1, -1), 0.5, colors.grey), ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"), ("ALIGN", (1, 1), (-1, -1), "CENTER")]))
        story.append(tbl)
        story.append(Spacer(1, 0.7 * cm))

    def _build_section_histograms(self, story, numeric_cols, hist_paths, styles):
        story.append(Paragraph("<b><font size=16 color='#457B9D'>Section 3: Histograms</font></b>", styles["SectionHeader"]))
        story.append(Spacer(1, 0.5 * cm))
        rows = []
        temp_row = []
        for i, col in enumerate(numeric_cols):
            if col not in hist_paths: continue
            block = self._image_with_caption(hist_paths[col], col, 7.5, 5, styles)
            temp_row.append(block)
            if len(temp_row) == 2:
                rows.append(temp_row)
                temp_row = []
        if temp_row:
            temp_row.append(Spacer(1, 1))
            rows.append(temp_row)
        hist_table = Table(rows, colWidths=[8 * cm, 8 * cm], hAlign="CENTER")
        hist_table.setStyle(TableStyle([("ALIGN", (0, 0), (-1, -1), "CENTER"), ("VALIGN", (0, 0), (-1, -1), "TOP")]))
        story.append(hist_table)
        story.append(PageBreak())

    def _build_section_time_series(self, story, numeric_cols, ts_paths, win, styles):
        story.append(Paragraph("<b><font size=16 color='#457B9D'>Section 4: Time-Series Trends</font></b>", styles["SectionHeader"]))
        story.append(Spacer(1, 0.5 * cm))
        for col in numeric_cols:
            if col not in ts_paths: continue
            story.append(Paragraph(f"<b><font size=13 color='#1D3557'>Trend of {col} over Time</font></b>", styles["SubHeader"]))
            story.append(Spacer(1, 0.3 * cm))
            img = Image(str(ts_paths[col]), width=17 * cm, height=7 * cm)
            cap = Paragraph(f"Figure: {col} raw vs. {win}-pt moving average", styles["BodyTextSmall"])
            story.append(KeepTogether([img, cap]))
            story.append(Spacer(1, 0.7 * cm))
        story.append(PageBreak())

    def _build_section_correlation(self, story, numeric_cols, scatter_matrix_path, corr_matrix_path, styles):
        if len(numeric_cols) < 2: return
        story.append(Paragraph("<b><font size=16 color='#457B9D'>Section 5: Correlation Analysis</font></b>", styles["SectionHeader"]))
        story.append(Spacer(1, 0.5 * cm))
        if scatter_matrix_path:
            story.append(Paragraph("<b><font size=13 color='#1D3557'>Scatterplot Matrix</font></b>", styles["SubHeader"]))
            story.append(Spacer(1, 0.3 * cm))
            story.append(Image(str(scatter_matrix_path), width=17 * cm, height=17 * cm))
            story.append(Spacer(1, 0.7 * cm))
        if corr_matrix_path:
            story.append(Paragraph("<b><font size=13 color='#1D3557'>Correlation Matrix Heatmap</font></b>", styles["SubHeader"]))
            story.append(Spacer(1, 0.3 * cm))
            story.append(KeepTogether([Image(str(corr_matrix_path), width=10 * cm, height=8 * cm)]))
        story.append(PageBreak())

    def _build_section_diurnal(self, story, numeric_cols, diurnal_paths, styles):
        story.append(Paragraph("<b><font size=16 color='#457B9D'>Section 6: Diurnal Patterns</font></b>", styles["SectionHeader"]))
        story.append(Spacer(1, 0.5 * cm))
        for col in numeric_cols:
            if col not in diurnal_paths: continue
            story.append(Paragraph(f"<b><font size=13 color='#1D3557'>Hourly Average of {col}</font></b>", styles["SubHeader"]))
            story.append(Spacer(1, 0.3 * cm))
            story.append(KeepTogether([Image(str(diurnal_paths[col]), width=12 * cm, height=5 * cm)]))
            story.append(Spacer(1, 0.7 * cm))
        story.append(PageBreak())

    def _build_section_anomalies(self, story, numeric_cols, anomalies_info, df, datetime_col, styles):
        story.append(Paragraph("<b><font size=16 color='#457B9D'>Section 8: Anomalies & Thresholds</font></b>", styles["SectionHeader"]))
        story.append(Spacer(1, 0.5 * cm))
        for col in numeric_cols:
            jumps = anomalies_info.get(col, [])
            if jumps:
                story.append(Paragraph(f"<b><font size=13 color='#1D3557'>Sudden Jumps in {col}</font></b>", styles["SubHeader"]))
                # UPDATED: Use self.params
                story.append(Paragraph(f"Values where |Delta {col}| > {self.params.anomaly_multiplier:.0f}x std (~ {df[col].std():.2f})", styles["BodyTextSmall"]))
                story.append(Spacer(1, 0.2 * cm))
                table_data = [["Timestamp", col, "Delta " + col]]
                for rec in jumps[:5]:
                    ts = rec[datetime_col].strftime("%Y-%m-%d %H:%M")
                    val = rec[col]
                    delta = rec[f"__delta_{col}__"]
                    table_data.append([ts, f"{val:.2f}", f"{delta:.2f}"])
                tbl = Table(table_data, colWidths=[5 * cm, 5 * cm, 5 * cm], hAlign="LEFT")
                tbl.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F1FAEE")), ("GRID", (0, 0), (-1, -1), 0.5, colors.grey), ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"), ("ALIGN", (1, 1), (-1, -1), "CENTER")]))
                story.append(tbl)
                story.append(Spacer(1, 0.5 * cm))
            else:
                story.append(Paragraph(f"No sudden jumps detected in {col}.", styles["BodyTextSmall"]))
                story.append(Spacer(1, 0.5 * cm))
        story.append(PageBreak())

    def _build_section_conclusions(self, story, numeric_cols, df, styles):
        story.append(Paragraph("<b><font size=16 color='#457B9D'>Section 9: Conclusions & Recommendations</font></b>", styles["SectionHeader"]))
        story.append(Spacer(1, 0.5 * cm))
        conclusions = []
        for col in numeric_cols:
            col_min, col_max, col_mean = df[col].min(), df[col].max(), df[col].mean()
            lower_warn, upper_warn = (None, None)
            # UPDATED: Use self.params
            if "temp" in col.lower():
                lower_warn, upper_warn = self.params.default_temp_threshold.min, self.params.default_temp_threshold.max
            elif "hum" in col.lower():
                lower_warn, upper_warn = self.params.default_hum_threshold.min, self.params.default_hum_threshold.max
            
            if lower_warn is not None and upper_warn is not None:
                if col_min < lower_warn or col_max > upper_warn:
                    conclusions.append(f"{col}: range {col_min:.1f}-{col_max:.1f} exceeds thresholds [{lower_warn}, {upper_warn}].")
                else:
                    conclusions.append(f"{col}: range {col_min:.1f}-{col_max:.1f} (within thresholds).")
            else:
                conclusions.append(f"{col}: range {col_min:.2f}-{col_max:.2f}, mean ~ {col_mean:.2f}.")
        for line in conclusions:
            story.append(Paragraph(line, styles["BodyTextSmall"]))
            story.append(Spacer(1, 0.3 * cm))
        story.append(PageBreak())

    def _build_section_appendix(self, story, df, numeric_cols, datetime_col, styles):
        story.append(Paragraph("<b><font size=16 color='#457B9D'>Appendix: Raw Data Preview</font></b>", styles["SectionHeader"]))
        story.append(Spacer(1, 0.5 * cm))
        preview_df = df[[datetime_col] + numeric_cols].head(10).copy()
        preview_df[datetime_col] = preview_df[datetime_col].dt.strftime("%Y-%m-%d %H:%M")
        table_data = [preview_df.columns.tolist()] + preview_df.values.tolist()
        col_widths = [(A4[0] - 3 * cm) / len(table_data[0])] * len(table_data[0])
        preview_tbl = Table(table_data, colWidths=col_widths, hAlign="LEFT")
        preview_tbl.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F1FAEE")), ("GRID", (0, 0), (-1, -1), 0.5, colors.grey), ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"), ("ALIGN", (1, 1), (-1, -1), "CENTER")]))
        story.append(preview_tbl)
        story.append(Spacer(1, 1 * cm))

    def _build_pdf(self, csv_name, logo_path, report_pdf, **kwargs):
        styles = self._get_report_styles()
        story = []
        self._append_cover_page(story, logo_path, csv_name, kwargs['date_min'], kwargs['date_max'], kwargs['numeric_cols'], styles)
        self._build_section_data_quality(story, kwargs['median_interval'], kwargs['missing_gaps'], kwargs['datetime_col'], styles)
        self._build_section_summary_stats(story, kwargs['numeric_cols'], kwargs['summary_stats'], styles)
        self._build_section_histograms(story, kwargs['numeric_cols'], kwargs['hist_paths'], styles)
        self._build_section_time_series(story, kwargs['numeric_cols'], kwargs['ts_paths'], kwargs['win'], styles)
        self._build_section_correlation(story, kwargs['numeric_cols'], kwargs['scatter_matrix_path'], kwargs['corr_matrix_path'], styles)
        self._build_section_diurnal(story, kwargs['numeric_cols'], kwargs['diurnal_paths'], styles)
        self._build_section_anomalies(story, kwargs['numeric_cols'], kwargs['anomalies_info'], kwargs['df'], kwargs['datetime_col'], styles)
        self._build_section_conclusions(story, kwargs['numeric_cols'], kwargs['df'], styles)
        self._build_section_appendix(story, kwargs['df'], kwargs['numeric_cols'], kwargs['datetime_col'], styles)
        
        doc = SimpleDocTemplate(str(report_pdf), pagesize=A4, rightMargin=1.5*cm, leftMargin=1.5*cm, topMargin=1.5*cm, bottomMargin=1.5*cm)
        doc.build(story, onFirstPage=lambda c,d: self._header_footer_callback(c,d,csv_name), onLaterPages=lambda c,d: self._header_footer_callback(c,d,csv_name))


# 3. THE ORCHESTRATION
@algorithm.validate
def validate(*args, **kwargs):
    try:
        _, file_path = next(algorithm.job_details.next_path())
        if not Path(file_path).exists():
            raise FileNotFoundError(f"Input file not found at {file_path}")
    except StopIteration:
        raise ValueError("No input files provided in job details.")

@algorithm.run
def run() -> str:
    params = algorithm.job_details.input_parameters
    _, csv_path_str = next(algorithm.job_details.next_path())
    csv_path = Path(csv_path_str)
    
    algorithm.logger.info(f"Processing {csv_path.name}")

    logo_path = csv_path.parent / "logo.png"
    if not logo_path.exists(): logo_path = None

    df = pd.read_csv(csv_path, dtype=str)

    # Use persistent temp location
    persistent_temp = Path("/tmp/report_gen")
    persistent_temp.mkdir(exist_ok=True, parents=True)

    with tempfile.TemporaryDirectory() as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        generator = PDFReportGenerator(params)
        
        pdf_path = generator.generate(
            df=df,
            csv_name=csv_path.name,
            logo_path=logo_path,
            output_dir=temp_dir
        )
        
        final_temp_path = persistent_temp / f"report_{datetime.now().timestamp()}.pdf"
        shutil.copy(pdf_path, final_temp_path)

    return str(final_temp_path)

@algorithm.save_results
def save(result: str, base: Path, *args, **kwargs):
    source_path = Path(result)
    output_file = base / "report.pdf"
    shutil.move(source_path, output_file)
    algorithm.logger.info(f"Report saved to {output_file}")

if __name__ == "__main__":
    algorithm()