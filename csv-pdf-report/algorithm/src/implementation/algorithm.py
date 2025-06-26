from logging import getLogger
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image,
    Table,
    TableStyle,
    PageBreak,
    KeepTogether,
)
from reportlab.lib import colors
from reportlab.lib.units import cm

from datetime import datetime

from oceanprotocol_job_details.ocean import JobDetails
from implementation.data import InputParameters

logger = getLogger(__name__)

class Algorithm:
    """
    Analyzes a CSV file with at least one datetime column and one or more numeric columns.
    Generates a PDF report with:
      - Cover Page
      - Executive Summary
      - Section 1: Data Quality
      - Section 2: Summary Statistics
      - Section 3: Histograms
      - Section 4: Time-Series Trends
      - Section 5: Correlation Analysis (if ≥2 numeric)
      - Section 6: Diurnal Patterns
      - Section 7: Forecast (linear + rolling avg)
      - Section 8: Anomalies
      - Section 9: Conclusions & Recommendations
      - Appendix: Raw Data Preview
    """

    def __init__(
        self,
        job_details: Optional[JobDetails[InputParameters]] = None,
    ):
        # Accepts a JobDetails object containing input file info and parameters
        self._job_details = job_details
        self.results: Optional[str] = None

    def _validate_input(self):
        # Ensure input files are present in job details
        if self._job_details:
            if not getattr(self._job_details, "files", None):
                raise ValueError("Ocean JobDetails contains no input files.")

    def run(self, temp_path: Path) -> "Algorithm":
        self._validate_input()

        # Get CSV file path from job details
        csv_path = self._job_details.files.files[0].input_files[0]

        if not csv_path.is_file():
            raise FileNotFoundError(f"CSV not found: {csv_path}")
        
        # Optional logo file in the same directory as CSV
        logo = csv_path.parent / "logo.png"
        logo_path = logo if logo.is_file() else None

        # Load CSV as strings, find datetime column, and sort by datetime
        df = pd.read_csv(csv_path, dtype=str)
        datetime_col = self._find_datetime_column(df)
        df[datetime_col] = pd.to_datetime(df[datetime_col], infer_datetime_format=True)
        df = df.sort_values(datetime_col).reset_index(drop=True)

        # Identify numeric columns
        numeric_cols = self._find_numeric_columns(df, datetime_col)
        if not numeric_cols:
            raise ValueError("No numeric columns found in CSV.")

        # Extract date range for report
        date_min = df[datetime_col].min().strftime("%Y-%m-%d %H:%M")
        date_max = df[datetime_col].max().strftime("%Y-%m-%d %H:%M")

        # Data quality: compute time intervals and detect large gaps
        df["__time_diff__"] = df[datetime_col].diff().dt.total_seconds().fillna(0)
        median_interval = df["__time_diff__"].median()
        missing_gaps = df[df["__time_diff__"] > 1.5 * median_interval][[datetime_col, "__time_diff__"]]

        # Summary statistics for numeric columns
        summary_stats = df[numeric_cols].describe()

        # Daily aggregates (not used in report, but available)
        daily_agg = (
            df.set_index(datetime_col).resample("D")[numeric_cols].agg(["min", "mean", "max"])
        )
        daily_agg.columns = [f"{col}_{stat}" for col, stat in daily_agg.columns.to_flat_index()]
        daily_agg = daily_agg.reset_index()
        daily_agg["date_str"] = daily_agg[datetime_col].dt.strftime("%Y-%m-%d")

        # Generate plots for report sections
        win = max(5, min(int(len(df) * 0.1), 50))
        hist_paths = self._generate_histograms(df, numeric_cols, temp_path)
        ts_paths = self._generate_time_series_plots(df, numeric_cols, datetime_col, win, temp_path)
        scatter_matrix_path, corr_matrix_path = self._generate_correlation_plots(df, numeric_cols, temp_path)
        diurnal_paths = self._generate_diurnal_plots(df, numeric_cols, datetime_col, temp_path)
        forecast_paths, forecast_summaries = self._generate_forecasts(df, numeric_cols, datetime_col, temp_path)
        anomalies_info = self._detect_anomalies(df, numeric_cols, datetime_col)
        
        # Build the PDF report
        self._build_pdf(
            csv_path=csv_path,
            logo_path=logo_path,
            report_pdf="temp_report.pdf",
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
            forecast_paths=forecast_paths,
            forecast_summaries=forecast_summaries,
            anomalies_info=anomalies_info,
            df=df,
            datetime_col=datetime_col,
            output_dir=temp_path,
        )

        self.results = str(temp_path / "temp_report.pdf")
        logger.info(f"PDF report written to {self.results}")
        return self

    def save_result(self, path: Path) -> None:
        """Save the generated PDF report to the specified path."""
        if self.results is None:
            logger.error("No results to save.")
            raise ValueError("No results to save.")
        
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Move the file to the final destination
        import shutil
        shutil.move(self.results, path)
        logger.info(f"Saved PDF report to {path}")

    # ────────────── Helper Methods ─────────────────────────────────────────────

    def _find_datetime_column(self, df: pd.DataFrame) -> str:
        """Return the first column that can be parsed as datetime."""
        for col in df.columns:
            try:
                pd.to_datetime(df[col], infer_datetime_format=True)
                return col
            except Exception:
                continue
        raise ValueError("No column could be parsed as datetime.")

    def _find_numeric_columns(self, df: pd.DataFrame, datetime_col: str) -> List[str]:
        """Return columns that can be coerced to numeric, excluding the datetime column."""
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

    def _generate_histograms(
        self, df: pd.DataFrame, numeric_cols: List[str], output_dir: Path
    ) -> Dict[str, Path]:
        """Generate and save a histogram for each numeric column."""
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

    def _generate_time_series_plots(
        self,
        df: pd.DataFrame,
        numeric_cols: List[str],
        datetime_col: str,
        win: int,
        output_dir: Path,
    ) -> Dict[str, Path]:
        """Plot raw and rolling average time series for each numeric column."""
        ts_paths: Dict[str, Path] = {}
        for col in numeric_cols:
            try:
                fig, ax = plt.subplots(figsize=(8, 3.5))
                ax.plot(df[datetime_col], df[col], label=f"{col} (raw)", color="tab:red", alpha=0.6)
                ax.plot(
                    df[datetime_col],
                    df[col].rolling(window=win, min_periods=1).mean(),
                    label=f"{col} {win}-pt MA",
                    color="tab:orange",
                    linestyle="--",
                )
                ax.set_xlabel("Time")
                ax.set_ylabel(col)
                ax.legend(fontsize=8)
                plt.tight_layout()

                # Format x-axis for dates
                import matplotlib.dates as mdates

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

    def _generate_correlation_plots(
        self, df: pd.DataFrame, numeric_cols: List[str], output_dir: Path
    ) -> Tuple[Optional[Path], Optional[Path]]:
        """
        If at least two numeric columns, generate:
          - Scatterplot matrix
          - Correlation heatmap
        """
        if len(numeric_cols) < 2:
            return None, None

        scatter_matrix_path, corr_matrix_path = None, None

        # Scatterplot matrix
        try:
            sns.set(style="whitegrid")
            g = sns.pairplot(df[numeric_cols].dropna(), kind="scatter", plot_kws={"s": 10, "alpha": 0.6})
            scatter_matrix_path = output_dir / "scatter_matrix.png"
            g.figure.savefig(scatter_matrix_path, dpi=120)
            plt.close(g.figure)
        except Exception as e:
            logger.warning(f"Could not create scatterplot matrix: {e}")

        # Correlation heatmap
        try:
            corr = df[numeric_cols].corr()
            fig, ax = plt.subplots(figsize=(4 + 0.5 * len(numeric_cols), 4))
            sns.heatmap(
                corr,
                annot=True,
                fmt=".2f",
                cmap="coolwarm",
                cbar_kws={"shrink": 0.5},
                ax=ax,
                linewidths=0.3,
            )
            ax.set_title("Correlation Matrix")
            plt.tight_layout()
            corr_matrix_path = output_dir / "corr_matrix.png"
            fig.savefig(corr_matrix_path, dpi=120)
            plt.close(fig)
        except Exception as e:
            logger.warning(f"Could not create correlation heatmap: {e}")

        return scatter_matrix_path, corr_matrix_path

    def _generate_diurnal_plots(
        self, df: pd.DataFrame, numeric_cols: List[str], datetime_col: str, output_dir: Path
    ) -> Dict[str, Path]:
        """Plot average value by hour of day for each numeric column."""
        df["__hour__"] = df[datetime_col].dt.hour
        diurnal_paths: Dict[str, Path] = {}
        for col in numeric_cols:
            try:
                hourly_avg = df.groupby("__hour__")[col].mean().reset_index()
                fig, ax = plt.subplots(figsize=(6, 2.8))
                ax.plot(hourly_avg["__hour__"], hourly_avg[col], marker="o", color="tab:green")
                ax.set_xticks(range(
                    self._job_details.input_parameters.diurnal_xticks.min,
                    self._job_details.input_parameters.diurnal_xticks.max,
                    self._job_details.input_parameters.diurnal_xticks.step
                ))
                ax.set_xlabel("Hour of Day")
                ax.set_ylabel(f"Avg {col}")
                ax.set_title(f"Diurnal Pattern of {col}")
                plt.tight_layout()

                p = output_dir / f"diurnal_{col}.png"
                fig.savefig(p, dpi=120)
                plt.close(fig)
                diurnal_paths[col] = p
            except Exception as e:
                logger.warning(f"Could not plot diurnal pattern for '{col}': {e}")
        return diurnal_paths

    def _generate_forecasts(
        self,
        df: pd.DataFrame,
        numeric_cols: List[str],
        datetime_col: str,
        output_dir: Path,
    ) -> Tuple[Dict[str, Path], Dict[str, str]]:
        """
        For each numeric column, if enough data:
          - Fit linear trend for 24h forecast
          - Use 6h rolling average as constant forecast
        Returns forecast plot paths and summary strings.
        """
        forecast_paths: Dict[str, Path] = {}
        forecast_summaries: Dict[str, str] = {}
        last_ts = df[datetime_col].max()
        future_times = [last_ts + pd.Timedelta(hours=i) for i in range(1, 25)]
        future_ord = np.array([t.toordinal() for t in future_times])

        for col in numeric_cols:
            try:
                df_hour = (
                    df[[datetime_col, col]]
                    .set_index(datetime_col)
                    .resample("h")[col]
                    .mean()
                    .ffill()
                )
                if df_hour.dropna().shape[0] < self._job_details.input_parameters.min_points_for_forecast:
                    logger.info(f"Skipping forecast for '{col}'—not enough data points.")
                    continue

                X = np.array([ts.toordinal() for ts in df_hour.index]).reshape(-1, 1)
                y = df_hour.values

                # Linear fit
                m, b = np.polyfit(X.flatten(), y, 1)
                lin_forecast = m * future_ord + b

                # 6h rolling avg
                roll_avg = df_hour.rolling(window=self._job_details.input_parameters.forecast_roll_window, min_periods=1).mean().iloc[-1]
                const_forecast = np.full(shape=len(future_times), fill_value=roll_avg)

                # Plot historical and forecasted values
                fig, ax = plt.subplots(figsize=(6, 2.8))
                hist_plot = df_hour[-72:]
                ax.plot(
                    hist_plot.index,
                    hist_plot.values,
                    label=f"{col} (last 72h)",
                    color="tab:blue",
                    alpha=0.6,
                )
                ax.plot(
                    future_times,
                    lin_forecast,
                    label="Linear 24h Forecast",
                    color="tab:red",
                    linestyle="--",
                )
                ax.plot(
                    future_times,
                    const_forecast,
                    label="6h MA Forecast",
                    color="tab:orange",
                    linestyle=":",
                )
                ax.set_xlabel("Time")
                ax.set_ylabel(col)
                ax.legend(fontsize=7)
                ax.set_title(f"Forecast for {col} (Next 24h)")
                plt.tight_layout()

                import matplotlib.dates as mdates
                locator = mdates.AutoDateLocator(minticks=3, maxticks=5)
                fmt = mdates.AutoDateFormatter(locator)
                ax.xaxis.set_major_locator(locator)
                ax.xaxis.set_major_formatter(fmt)
                fig.autofmt_xdate(rotation=25)

                p = output_dir / f"forecast_{col}.png"
                fig.savefig(p, dpi=120)
                plt.close(fig)

                forecast_paths[col] = p
                change_24h = (m * future_ord[-1] + b) - (m * future_ord[0] + b)
                forecast_summaries[col] = (
                    f"Linear trend: slope={m:.3f}, intercept={b:.3f} → "
                    f"change over 24h ≈ {change_24h:.2f}."
                )
            except Exception as e:
                logger.warning(f"Could not generate forecast for '{col}': {e}")

        return forecast_paths, forecast_summaries

    def _detect_anomalies(
        self, df: pd.DataFrame, numeric_cols: List[str], datetime_col: str
    ) -> Dict[str, List[dict[str, Any]]]:
        """
        Flag rows where the absolute change in value exceeds a threshold.
        Returns a dict mapping column → list of anomaly records.
        """
        anomalies_info: Dict[str, List[dict[str, Any]]] = {}
        for col in numeric_cols:
            col_std = df[col].std()
            df[f"__delta_{col}__"] = df[col].diff().abs().fillna(0)
            threshold = self._job_details.input_parameters.anomaly_multiplier * col_std
            jumps = df[df[f"__delta_{col}__"] > threshold][[datetime_col, col, f"__delta_{col}__"]]
            anomalies_info[col] = jumps.to_dict(orient="records")
        return anomalies_info

    def _get_report_styles(self) -> dict[str, ParagraphStyle]:
        """
        Return a stylesheet dictionary customized for the PDF report.
        """
        styles = getSampleStyleSheet()

        styles["Title"].fontName = "Helvetica-Bold"
        styles["BodyText"].fontName = "Helvetica"

        styles.add(
            ParagraphStyle(
                name="SectionHeader",
                fontSize=14,
                fontName="Helvetica-Bold",
                textColor=colors.HexColor("#457B9D"),
                spaceAfter=6,
            )
        )
        styles.add(
            ParagraphStyle(
                name="SubHeader",
                fontSize=12,
                fontName="Helvetica-Bold",
                textColor=colors.HexColor("#1D3557"),
                spaceAfter=4,
            )
        )
        styles.add(ParagraphStyle(name="BodyTextSmall", fontSize=9, fontName="Helvetica", leading=11))

        return styles

    def _image_with_caption(
        self,
        img_path: Path,
        caption_text: str,
        width_cm: float,
        height_cm: float,
        styles: dict[str, ParagraphStyle],
    ) -> Table:
        """
        Return a table with an image and its caption, centered and padded.
        """
        img = Image(str(img_path), width=width_cm * cm, height=height_cm * cm)
        caption = Paragraph(caption_text, styles["BodyTextSmall"])
        tbl = Table([[img], [caption]], colWidths=[width_cm * cm], hAlign="CENTER")
        tbl.setStyle(
            TableStyle(
                [
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("BOTTOMPADDING", (0, 0), (0, 0), 4),
                ]
            )
        )
        return tbl

    def _append_cover_page(
        self,
        story: list[Any],
        logo_path: Optional[Path],
        csv_name: str,
        date_min: str,
        date_max: str,
        numeric_cols: List[str],
        styles: dict[str, ParagraphStyle],
    ) -> None:
        """
        Add cover page with logo, title, date range, column list, timestamp, and confidentiality notice.
        """
        if logo_path:
            story.append(Image(str(logo_path), width=6 * cm, height=6 * cm))
            story.append(Spacer(1, 1.5 * cm))

        story.append(
            Paragraph(
                f"<b><font size=24 color='#1D3557'>Data Report: {csv_name}</font></b>",
                styles["Title"],
            )
        )
        story.append(Spacer(1, 0.7 * cm))
        story.append(
            Paragraph(
                f"<font size=12 color='#457B9D'>Date Range: {date_min} – {date_max}</font>",
                styles["BodyText"],
            )
        )
        story.append(Spacer(1, 0.3 * cm))
        story.append(
            Paragraph(
                f"<font size=12 color='#457B9D'>Numeric Columns ({len(numeric_cols)}): {', '.join(numeric_cols)}</font>",
                styles["BodyText"],
            )
        )
        story.append(Spacer(1, 0.5 * cm))
        story.append(
            Paragraph(
                f"<font size=10 color='#A8A8A8'>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}</font>",
                styles["BodyTextSmall"],
            )
        )
        story.append(Spacer(1, 2 * cm))
        story.append(
            Paragraph("<font size=9 color='#A8A8A8'>Confidential – Internal Use Only</font>", styles["BodyTextSmall"])
        )
        story.append(PageBreak())

    def _header_footer_callback(self, canvas, doc, csv_name: str) -> None:
        """
        Draw header (skipped on page 1) and footer on all pages.
        Header: "Data Report: {csv_name}"
        Footer: centered page number, right-aligned confidentiality notice.
        """
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
        canvas.drawRightString(A4[0] - 1.5 * cm, 1 * cm, "Confidential – Internal Use Only")
        canvas.restoreState()

    def _build_pdf(
        self,
        csv_path: Path,
        logo_path: Optional[Path],
        report_pdf: str,
        date_min: str,
        date_max: str,
        numeric_cols: List[str],
        median_interval: float,
        missing_gaps: pd.DataFrame,
        summary_stats: pd.DataFrame,
        hist_paths: Dict[str, Path],
        ts_paths: Dict[str, Path],
        win: int,
        scatter_matrix_path: Optional[Path],
        corr_matrix_path: Optional[Path],
        diurnal_paths: Dict[str, Path],
        forecast_paths: Dict[str, Path],
        forecast_summaries: Dict[str, str],
        anomalies_info: Dict[str, List[dict[str, Any]]],
        df: pd.DataFrame,
        datetime_col: str,
        output_dir: Path,
    ) -> None:
        """Assemble all report sections and generate the PDF."""
        styles = self._get_report_styles()
        story: list[Any] = []

        # Cover Page
        self._append_cover_page(
            story, 
            logo_path, 
            csv_path.name, 
            date_min, 
            date_max, 
            numeric_cols, 
            styles
        )

        # Add all report sections
        self._build_section_data_quality(
            story, median_interval, missing_gaps, datetime_col, styles
        )
        self._build_section_summary_stats(story, numeric_cols, summary_stats, styles)
        self._build_section_histograms(story, numeric_cols, hist_paths, styles)
        self._build_section_time_series(story, numeric_cols, ts_paths, win, styles)
        self._build_section_correlation(story, numeric_cols, scatter_matrix_path, corr_matrix_path, styles)
        self._build_section_diurnal(story, numeric_cols, diurnal_paths, styles)
        self._build_section_forecast(story, numeric_cols, forecast_paths, forecast_summaries, styles)
        self._build_section_anomalies(story, numeric_cols, anomalies_info, df, datetime_col, styles)
        self._build_section_conclusions(story, numeric_cols, df, styles)
        self._build_section_appendix(story, df, numeric_cols, datetime_col, styles)

        # Build PDF with header/footer
        doc = SimpleDocTemplate(
            str(output_dir / report_pdf),
            pagesize=A4,
            rightMargin=1.5 * cm,
            leftMargin=1.5 * cm,
            topMargin=1.5 * cm,
            bottomMargin=1.5 * cm,
        )
        doc.build(
            story,
            onFirstPage=lambda c, d: self._header_footer_callback(c, d, csv_path.name),
            onLaterPages=lambda c, d: self._header_footer_callback(c, d, csv_path.name),
        )
