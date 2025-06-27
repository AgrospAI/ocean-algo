from logging import getLogger
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import os
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
    Examines any CSV with:
      1. At least one column convertible to datetime,
      2. One or more numeric columns (int/float).
    Produces a PDF with:
      - Cover Page
      - Executive Summary
      - Section 1: Data Quality
      - Section 2: Summary Statistics
      - Section 3: Histograms (2 per row)
      - Section 4: Time-Series Trends
      - Section 5: Correlation Analysis (if ≥2 numeric)
      - Section 6: Diurnal Patterns
      - Section 7: Forecast (linear + rolling avg)
      - Section 8: Anomalies (|Δ| > ANOMALY_MULTIPLIER × std)
      - Section 9: Conclusions & Recommendations
      - Appendix: Raw Data Preview (first 10 rows)
    """

    def __init__(
        self,
        job_details: Optional[JobDetails[InputParameters]] = None,
    ):
        # You must supply exactly one of these:
        self._job_details = job_details
        self.results: Optional[str] = None

    def _validate_input(self):
        if self._job_details:
            if not getattr(self._job_details, "files", None):
                raise ValueError("Ocean JobDetails contains no input files.")
            
    def run(self, temp_path: Path) -> "Algorithm":
        self._validate_input()

        # 1) Determine CSV path
        csv_path = self._job_details.files.files[0].input_files[0]
        if not csv_path.is_file():
            raise FileNotFoundError(f"CSV not found: {csv_path}")
        
        # 3) Optional logo alongside CSV
        logo = csv_path.parent / "logo.png"
        logo_path = logo if logo.is_file() else None

        # 4) Load & parse CSV
        df = pd.read_csv(csv_path, dtype=str)
        datetime_col = self._find_datetime_column(df)
        df[datetime_col] = pd.to_datetime(df[datetime_col])
        df = df.sort_values(datetime_col).reset_index(drop=True)

        # 4a) Numeric columns
        numeric_cols = self._find_numeric_columns(df, datetime_col)
        if not numeric_cols:
            raise ValueError("No numeric columns found in CSV.")

        # 5) Metadata
        date_min = df[datetime_col].min().strftime("%Y-%m-%d %H:%M")
        date_max = df[datetime_col].max().strftime("%Y-%m-%d %H:%M")

        # 6) Data Quality → time diffs
        df["__time_diff__"] = df[datetime_col].diff().dt.total_seconds().fillna(0)
        median_interval = df["__time_diff__"].median()
        missing_gaps = df[df["__time_diff__"] > 1.5 * median_interval][[datetime_col, "__time_diff__"]]

        # 7) Summary statistics
        summary_stats = df[numeric_cols].describe()

        # 8) Daily aggregates (only used if you decide to include them later)
        daily_agg = (
            df.set_index(datetime_col).resample("D")[numeric_cols].agg(["min", "mean", "max"])
        )
        daily_agg.columns = [f"{col}_{stat}" for col, stat in daily_agg.columns.to_flat_index()]
        daily_agg = daily_agg.reset_index()
        daily_agg["date_str"] = daily_agg[datetime_col].dt.strftime("%Y-%m-%d")

        # 9) Generate all plots
        win = max(5, min(int(len(df) * 0.1), 50))
        hist_paths = self._generate_histograms(df, numeric_cols, temp_path)
        ts_paths = self._generate_time_series_plots(df, numeric_cols, datetime_col, win, temp_path)
        scatter_matrix_path, corr_matrix_path = self._generate_correlation_plots(df, numeric_cols, temp_path)
        diurnal_paths = self._generate_diurnal_plots(df, numeric_cols, datetime_col, temp_path)
        forecast_paths, forecast_summaries = self._generate_forecasts(df, numeric_cols, datetime_col, temp_path)
        anomalies_info = self._detect_anomalies(df, numeric_cols, datetime_col)
        
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
        """Save the PDF report directly to the specified path."""
        if self.results is None:
            logger.error("No results to save.")
            raise ValueError("No results to save.")
        
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Copy the file directly to the final destination
        import shutil
        shutil.copy(self.results, path)
        os.remove(self.results)  # Clean up the temporary file
        logger.info(f"Saved PDF report to {path}")

    # ────────────── Helper Methods ─────────────────────────────────────────────

    def _find_datetime_column(self, df: pd.DataFrame) -> str:
        """Return the name of the first column that can be parsed to datetime."""
        for col in df.columns:
            try:
                pd.to_datetime(df[col])
                return col
            except Exception:
                continue
        raise ValueError("No column could be parsed as datetime.")

    def _find_numeric_columns(self, df: pd.DataFrame, datetime_col: str) -> List[str]:
        """Return a list of columns that can be coerced to numeric (excluding datetime_col)."""
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
        """Generate one histogram per numeric column; return map col→Path."""
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
        """Plot raw vs. rolling average for each numeric column; return map col→Path."""
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

                # Format x-axis nicely
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
        If ≥2 numeric columns:
          - Pairplot → scatter_matrix.png
          - Correlation heatmap → corr_matrix.png
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
        """Compute hour-of-day average for each numeric column and plot; return col→Path."""
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
        Resample each column to hourly. If fewer than MIN_POINTS_FOR_FORECAST points,
        skip. Otherwise:
          - Fit linear trend on ordinal() dates → linear 24h forecast
          - Take 6h rolling avg → constant forecast
        Return (col→forecast_png, col→summary_str).
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

                # Plot
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
        Flags any Δcol > ANOMALY_MULTIPLIER × std(col).
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
        Return a stylesheet dictionary customized for our PDF.
        - Title: Helvetica-Bold, size 24
        - BodyText: Helvetica, size 12
        - SectionHeader: Helvetica-Bold, size 14, color #457B9D
        - SubHeader: Helvetica-Bold, size 12, color #1D3557
        - BodyTextSmall: Helvetica, size 9
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
        Return a one-column Table containing:
           ┌─────────────┐
           │  <Image>    │  ← width_cm × height_cm
           │ "caption"   │
           └─────────────┘
        Centered and padded.
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
        Append:
          [logo?], Title, DateRange, ColumnList, timestamp, confidentiality → page break.
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
        Draw header (skipped on page 1) and footer on all pages:
          - Header: "Data Report: {csv_name}"
          - Footer: centered page number, right-aligned "Confidential – Internal Use Only"
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

    def _build_section_data_quality(
        self,
        story: list[Any],
        median_interval: float,
        missing_gaps: pd.DataFrame,
        datetime_col: str,
        styles: dict[str, ParagraphStyle],
    ) -> None:
        """Append Section 1: Data Quality (median interval + list of large gaps) to story."""
        story.append(Paragraph("<b><font size=16 color='#457B9D'>Section 1: Data Quality</font></b>", styles["SectionHeader"]))
        story.append(Spacer(1, 0.5 * cm))
        story.append(Paragraph(f"Median sampling interval: {median_interval:.1f} seconds", styles["BodyTextSmall"]))
        story.append(Spacer(1, 0.2 * cm))
        if not missing_gaps.empty:
            story.append(Paragraph("<b><font size=13 color='#1D3557'>Large Sampling Gaps (time → gap_seconds):</font></b>", styles["SubHeader"]))
            for _, row in missing_gaps.iterrows():
                ts_str = row[datetime_col].strftime("%Y-%m-%d %H:%M:%S")
                gap = row["__time_diff__"]
                story.append(Paragraph(f"{ts_str} → gap = {gap:.1f} s", styles["BodyTextSmall"]))
        else:
            story.append(Paragraph("No large sampling gaps detected.", styles["BodyTextSmall"]))
        story.append(PageBreak())

    def _build_section_summary_stats(
        self,
        story: list[Any],
        numeric_cols: List[str],
        summary_stats: pd.DataFrame,
        styles: dict[str, ParagraphStyle],
    ) -> None:
        """Append Section 2: Summary Statistics (table of min/25%/50%/mean/75%/max/std)."""
        story.append(Paragraph("<b><font size=16 color='#457B9D'>Section 2: Summary Statistics</font></b>", styles["SectionHeader"]))
        story.append(Spacer(1, 0.5 * cm))

        metrics = ["min", "25%", "50%", "mean", "75%", "max", "std"]
        table_data = [["Metric"] + numeric_cols]
        for m in metrics:
            row = [m.capitalize()] + [f"{summary_stats.at[m, col]:.2f}" for col in numeric_cols]
            table_data.append(row)

        col_width = (A4[0] - 3 * cm) / (len(numeric_cols) + 1)
        tbl = Table(table_data, colWidths=[col_width] * (len(numeric_cols) + 1), hAlign="CENTER")
        tbl.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F1FAEE")),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("ALIGN", (1, 1), (-1, -1), "CENTER"),
                ]
            )
        )
        story.append(tbl)
        story.append(Spacer(1, 0.7 * cm))

    def _build_section_histograms(
        self,
        story: list[Any],
        numeric_cols: List[str],
        hist_paths: Dict[str, Path],
        styles: dict[str, ParagraphStyle],
    ) -> None:
        """Append Section 3: Histograms (2 per row) to the story."""
        story.append(Paragraph("<b><font size=16 color='#457B9D'>Section 3: Histograms</font></b>", styles["SectionHeader"]))
        story.append(Spacer(1, 0.5 * cm))
        story.append(Paragraph("<b><font size=13 color='#1D3557'>Histograms of Numeric Columns</font></b>", styles["SubHeader"]))
        story.append(Spacer(1, 0.3 * cm))

        rows: List[List[Any]] = []
        temp_row: List[Any] = []
        for i, col in enumerate(numeric_cols):
            block = self._image_with_caption(
                img_path=hist_paths[col],
                caption_text=col,
                width_cm=7.5,
                height_cm=5,
                styles=styles,
            )
            temp_row.append(block)
            if len(temp_row) == 2:
                rows.append(temp_row)
                temp_row = []
        if temp_row:
            temp_row.append(Spacer(1, 1))
            rows.append(temp_row)

        hist_table = Table(rows, colWidths=[8 * cm, 8 * cm], hAlign="CENTER")
        hist_table.setStyle(
            TableStyle(
                [
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ]
            )
        )
        story.append(hist_table)
        story.append(PageBreak())

    def _build_section_time_series(
        self,
        story: list[Any],
        numeric_cols: List[str],
        ts_paths: Dict[str, Path],
        win: int,
        styles: dict[str, ParagraphStyle],
    ) -> None:
        """Append Section 4: Time-Series Trends (one plot per numeric column)."""
        story.append(Paragraph("<b><font size=16 color='#457B9D'>Section 4: Time-Series Trends</font></b>", styles["SectionHeader"]))
        story.append(Spacer(1, 0.5 * cm))
        for col in numeric_cols:
            story.append(Paragraph(f"<b><font size=13 color='#1D3557'>Trend of {col} over Time</font></b>", styles["SubHeader"]))
            story.append(Spacer(1, 0.3 * cm))
            img = Image(str(ts_paths[col]), width=17 * cm, height=7 * cm)
            cap = Paragraph(f"Figure: {col} raw vs. {win}-pt moving average", styles["BodyTextSmall"])
            story.append(KeepTogether([img, cap]))
            story.append(Spacer(1, 0.7 * cm))
        story.append(PageBreak())

    def _build_section_correlation(
        self,
        story: list[Any],
        numeric_cols: List[str],
        scatter_matrix_path: Optional[Path],
        corr_matrix_path: Optional[Path],
        styles: dict[str, ParagraphStyle],
    ) -> None:
        """Append Section 5: Correlation Analysis if ≥2 numeric columns."""
        if len(numeric_cols) < 2:
            return

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

    def _build_section_diurnal(
        self,
        story: list[Any],
        numeric_cols: List[str],
        diurnal_paths: Dict[str, Path],
        styles: dict[str, ParagraphStyle],
    ) -> None:
        """Append Section 6: Diurnal Patterns (hourly average) to story."""
        story.append(Paragraph("<b><font size=16 color='#457B9D'>Section 6: Diurnal Patterns</font></b>", styles["SectionHeader"]))
        story.append(Spacer(1, 0.5 * cm))
        for col in numeric_cols:
            story.append(Paragraph(f"<b><font size=13 color='#1D3557'>Hourly Average of {col}</font></b>", styles["SubHeader"]))
            story.append(Spacer(1, 0.3 * cm))
            story.append(KeepTogether([Image(str(diurnal_paths[col]), width=12 * cm, height=5 * cm)]))
            story.append(Spacer(1, 0.7 * cm))
        story.append(PageBreak())

    def _build_section_forecast(
        self,
        story: list[Any],
        numeric_cols: List[str],
        forecast_paths: Dict[str, Path],
        forecast_summaries: Dict[str, str],
        styles: dict[str, ParagraphStyle],
    ) -> None:
        """Append Section 7: Forecasting Next 24h to story."""
        story.append(Paragraph("<b><font size=16 color='#457B9D'>Section 7: Forecasting (Next 24 Hours)</font></b>", styles["SectionHeader"]))
        story.append(Spacer(1, 0.5 * cm))
        for col in numeric_cols:
            if col not in forecast_paths:
                continue  # skipped if not enough points
            story.append(Paragraph(f"<b><font size=13 color='#1D3557'>{col} Forecast</font></b>", styles["SubHeader"]))
            story.append(Spacer(1, 0.3 * cm))
            story.append(KeepTogether([Image(str(forecast_paths[col]), width=17 * cm, height=7 * cm)]))
            story.append(Spacer(1, 0.2 * cm))
            story.append(Paragraph(forecast_summaries[col], styles["BodyTextSmall"]))
            story.append(Spacer(1, 0.7 * cm))
        story.append(PageBreak())

    def _build_section_anomalies(
        self,
        story: list[Any],
        numeric_cols: List[str],
        anomalies_info: Dict[str, List[dict[str, Any]]],
        df: pd.DataFrame,
        datetime_col: str,
        styles: dict[str, ParagraphStyle],
    ) -> None:
        """Append Section 8: Anomalies & Thresholds to story."""
        story.append(Paragraph("<b><font size=16 color='#457B9D'>Section 8: Anomalies & Thresholds</font></b>", styles["SectionHeader"]))
        story.append(Spacer(1, 0.5 * cm))

        for col in numeric_cols:
            jumps = anomalies_info.get(col, [])
            if jumps:
                story.append(Paragraph(f"<b><font size=13 color='#1D3557'>Sudden Jumps in {col}</font></b>", styles["SubHeader"]))
                story.append(
                    Paragraph(
                        f"Values where |Δ{col}| > {self._job_details.input_parameters.anomaly_multiplier:.0f}× std (≈ {df[col].std():.2f})",
                        styles["BodyTextSmall"],
                    )
                )
                story.append(Spacer(1, 0.2 * cm))

                # Build a small table of the first 5 anomalies
                table_data = [["Timestamp", col, "Δ" + col]]
                for rec in jumps[:5]:
                    ts = rec[datetime_col].strftime("%Y-%m-%d %H:%M")
                    val = rec[col]
                    delta = rec[f"__delta_{col}__"]
                    table_data.append([ts, f"{val:.2f}", f"{delta:.2f}"])
                tbl = Table(table_data, colWidths=[5 * cm, 5 * cm, 5 * cm], hAlign="LEFT")
                tbl.setStyle(
                    TableStyle(
                        [
                            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F1FAEE")),
                            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                            ("ALIGN", (1, 1), (-1, -1), "CENTER"),
                        ]
                    )
                )
                story.append(tbl)
                story.append(Spacer(1, 0.5 * cm))
            else:
                story.append(Paragraph(f"No sudden jumps detected in {col}.", styles["BodyTextSmall"]))
                story.append(Spacer(1, 0.5 * cm))

        story.append(PageBreak())

    def _build_section_conclusions(
        self,
        story: list[Any],
        numeric_cols: List[str],
        df: pd.DataFrame,
        styles: dict[str, ParagraphStyle],
    ) -> None:
        """Append Section 9: Conclusions & Recommendations to story."""
        story.append(Paragraph("<b><font size=16 color='#457B9D'>Section 9: Conclusions & Recommendations</font></b>", styles["SectionHeader"]))
        story.append(Spacer(1, 0.5 * cm))

        conclusions: List[str] = []
        for col in numeric_cols:
            col_min = df[col].min()
            col_max = df[col].max()
            col_mean = df[col].mean()

            lower_warn, upper_warn = (None, None)
            if "temp" in col.lower():
                lower_warn = self._job_details.input_parameters.default_temp_threshold.min
                upper_warn = self._job_details.input_parameters.default_temp_threshold.max
            elif "hum" in col.lower():
                lower_warn = self._job_details.input_parameters.default_hum_threshold.min
                upper_warn = self._job_details.input_parameters.default_hum_threshold.max

            if lower_warn is not None and upper_warn is not None:
                if col_min < lower_warn or col_max > upper_warn:
                    conclusions.append(
                        f"{col}: range {col_min:.1f}–{col_max:.1f} exceeds thresholds [{lower_warn}, {upper_warn}]."
                    )
                else:
                    conclusions.append(
                        f"{col}: range {col_min:.1f}–{col_max:.1f} (within thresholds)."
                    )
            else:
                conclusions.append(
                    f"{col}: range {col_min:.2f}–{col_max:.2f}, mean ≈ {col_mean:.2f}."
                )

        for line in conclusions:
            story.append(Paragraph(line, styles["BodyTextSmall"]))
            story.append(Spacer(1, 0.3 * cm))

        story.append(Spacer(1, 0.5 * cm))
        story.append(
            Paragraph(
                f"Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                styles["BodyTextSmall"],
            )
        )
        story.append(PageBreak())

    def _build_section_appendix(
        self,
        story: list[Any],
        df: pd.DataFrame,
        numeric_cols: List[str],
        datetime_col: str,
        styles: dict[str, ParagraphStyle],
    ) -> None:
        """Append Appendix: first 10 rows of data, properly formatted."""
        story.append(Paragraph("<b><font size=16 color='#457B9D'>Appendix: Raw Data Preview</font></b>", styles["SectionHeader"]))
        story.append(Spacer(1, 0.5 * cm))

        preview_df = df[[datetime_col] + numeric_cols].head(10).copy()
        preview_df[datetime_col] = preview_df[datetime_col].dt.strftime("%Y-%m-%d %H:%M")

        table_data = [preview_df.columns.tolist()] + preview_df.values.tolist()
        col_widths = [(A4[0] - 3 * cm) / len(table_data[0])] * len(table_data[0])
        preview_tbl = Table(table_data, colWidths=col_widths, hAlign="LEFT")
        preview_tbl.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F1FAEE")),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("ALIGN", (1, 1), (-1, -1), "CENTER"),
                ]
            )
        )
        story.append(preview_tbl)
        story.append(Spacer(1, 1 * cm))

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
        """Assemble all sections into a multi-page PDF via ReportLab."""
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

        # Sections
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