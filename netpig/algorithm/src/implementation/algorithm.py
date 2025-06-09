import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from logging import getLogger
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph, Frame, Table, TableStyle
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib import colors
from reportlab.lib.units import inch, cm
from reportlab.lib.utils import ImageReader
from datetime import datetime
from io import BytesIO
from oceanprotocol_job_details.ocean import JobDetails

logger = getLogger(__name__)

class Algorithm:
    def __init__(self, job_details: JobDetails):
        self._job_details = job_details
        self.results: Optional[_ResultType] = None
        self.logo_path = "src/img/netpig-logo.png"

        self.thresholds = {
            "CO2": {"recommended": "3000 ppm", "alert": "> 6000 ppm"},
            "NH3": {"recommended": "10 ppm", "alert": "> 20 ppm"}
        }

    def _validate_input(self) -> None:
        if not self._job_details.files:
            logger.warning("No files found")
            raise ValueError("No files found")
        
    def run(self) -> "Algorithm":
        self._validate_input()

        try:
            self.df = pd.read_csv(self._job_details.files.files[0].input_files[0], sep=None, engine="python")
            self.df.columns = self.df.columns.str.strip()
            self.df["CO2"] = pd.to_numeric(self.df["CO2"], errors="coerce")
            self.df["NH3"] = pd.to_numeric(self.df["NH3"], errors="coerce")
            self.results = self.df
        except Exception as e:
            logger.error(f"Error reading input file: {e}")
            self.df = None
            self.results = None
            raise
        return self
    
        # self.results = "ALGO RESULTS"
        # return self

    def save_result(self, output_path: Path) -> None:
        """Generate the PDF report with all required sections and graphics."""
        if self.df is None:
            logger.error("No data loaded. Cannot save result.")
            return
        c = canvas.Canvas(str(output_path / "report.pdf"), pagesize=A4)
        width, height = A4
        left_margin, right_margin, top_margin = 70, 70, height - 70
        frame_width = width - left_margin - right_margin
        img_width = width - left_margin - right_margin
        img_height = 180

        self._add_logo(c, width, height)
        self._add_title(c, left_margin, top_margin)
        self._add_intro(c, left_margin, top_margin, frame_width)
        current_y = self._add_thresholds_section_full(c, left_margin, top_margin, frame_width, self.thresholds, width)
        current_y = self._add_evolution_section(c, left_margin, right_margin, top_margin, width, self.df, img_width, img_height)
        self._add_distribution_section(c, left_margin, width, height, self.df, img_width, img_height)
        current_y = self._add_summary_section(c, left_margin, top_margin, width, self.df)
        current_y = self._add_diagnosis(c, self.df, current_y, width, left_margin)
        self._add_footer(c, width, right_margin)
        c.save()

    def _add_logo(self, c, width, height):
        if self.logo_path and Path(self.logo_path).is_file():
            try:
                logo = ImageReader(str(self.logo_path))
                logo_width = 0.7 * inch
                logo_height = 0.7 * inch
                c.drawImage(logo, width - logo_width - 10, height - logo_height - 10, width=logo_width, height=logo_height, preserveAspectRatio=True, mask='auto')
            except Exception as e:
                logger.warning(f"Could not add logo to PDF: {e}")

    def _add_title(self, c, left_margin, top_margin):
        c.setFont("Helvetica-Bold", 25)
        c.drawString(left_margin, top_margin, "Sustainability Report")

    def _add_intro(self, c, left_margin, top_margin, frame_width):
        style = ParagraphStyle(name="Justified", fontName="Helvetica", fontSize=10, leading=16, alignment=TA_JUSTIFY)
        intro_text = (
            "This document presents an environmental analysis based on data collected from the facility. "
            "The main objective is to assess whether the farm meets the air quality requirements necessary to "
            "obtain a sustainability certification, based on predefined thresholds for key indicators."
        )
        paragraph = Paragraph(intro_text, style)
        frame_height = 100
        frame = Frame(left_margin, top_margin - 130, frame_width, frame_height, showBoundary=0)
        frame.addFromList([paragraph], c)

    def _add_thresholds_section_full(self, c, left_margin, top_margin, frame_width, thresholds, width):
        current_y = top_margin - 120
        c.setFont("Helvetica-Bold", 14)
        c.drawString(left_margin, current_y, "Defined Limits for Certification")
        current_y -= 10
        style_small = ParagraphStyle(name="SmallJustified", fontName="Helvetica", fontSize=9, leading=12, alignment=TA_JUSTIFY)
        cert_text = ("To obtain the certification, the farm must maintain pollutant levels within the recommended limits and must never exceed the alert values.")
        paragraph_cert = Paragraph(cert_text, style_small)
        frame_cert = Frame(left_margin, current_y - 40, frame_width, 40, showBoundary=0)
        frame_cert.addFromList([paragraph_cert], c)
        current_y -= 60
        current_y = self._add_thresholds_section(c, thresholds, current_y, width, left_margin)
        return current_y

    def _add_evolution_section(self, c, left_margin, right_margin, top_margin, width, df, img_width, img_height):
        current_y = top_margin - 320
        c.setFont("Helvetica-Bold", 14)
        c.drawString(left_margin, current_y, "Indicators Evolution")
        current_y -= 10
        buf1 = self._plot_simple_evolution(df, 'CO2', 'CO2 Evolution')
        buf2 = self._plot_simple_evolution(df, 'NH3', 'NH3 Evolution')
        img1 = ImageReader(buf1)
        img2 = ImageReader(buf2)
        c.drawImage(img1, left_margin, current_y - img_height, width=img_width, height=img_height)
        current_y -= img_height + 30
        c.drawImage(img2, left_margin, current_y - img_height, width=img_width, height=img_height)
        current_y -= img_height + 30
        c.setFont("Helvetica", 9)
        c.drawCentredString(width / 2, 30, "1")
        c.showPage()
        return current_y

    def _add_distribution_section(self, c, left_margin, width, height, df, img_width, img_height):
        current_y = height - 70
        c.setFont("Helvetica-Bold", 14)
        c.drawString(left_margin, current_y, "Data Distribution and Outliers")
        current_y -= 20
        buf_dist_co2 = self._plot_distribution(df, 'CO2', 'CO2 Histogram', 'CO2 Boxplot')
        buf_dist_nh3 = self._plot_distribution(df, 'NH3', 'NH3 Histogram', 'NH3 Boxplot')
        c.drawImage(ImageReader(buf_dist_co2), left_margin, current_y - img_height, width=img_width, height=img_height)
        current_y -= img_height + 30
        c.drawImage(ImageReader(buf_dist_nh3), left_margin, current_y - img_height, width=img_width, height=img_height)
        current_y -= img_height + 30
        c.setFont("Helvetica", 9)
        c.drawCentredString(width / 2, 30, "3")

    def _add_summary_section(self, c, left_margin, top_margin, width, df):
        current_y = top_margin - 450
        c.setFont("Helvetica-Bold", 14)
        c.drawString(left_margin, current_y, "Environmental Quality Summary")
        current_y -= 30
        current_y = self._add_summary_table(c, df, left_margin, current_y, width)
        return current_y

    def _add_footer(self, c, width, right_margin):
        issue_date = datetime.today().strftime("Report issued on %B %d, %Y")
        c.setFont("Helvetica-Oblique", 9)
        c.setFillColor(colors.grey)
        c.drawRightString(width - right_margin, 40, issue_date)
        c.setFillColor(colors.black)
        c.setFont("Helvetica", 9)
        c.drawCentredString(width / 2, 30, "2")

    def _add_thresholds_section(self, c, thresholds, current_y, width, left_margin):
        c.setFont("Helvetica-Bold", 11)
        for gas, values in thresholds.items():
            c.drawString(left_margin, current_y, f"{gas}:")
            current_y -= 15
            c.setFont("Helvetica", 10)
            c.drawString(left_margin + 15, current_y, f"• Recommended limit: {values.get('recommended', '')}")
            current_y -= 15
            c.drawString(left_margin + 15, current_y, f"• High value (alert): {values.get('alert', '')}")
            current_y -= 25
            c.setFont("Helvetica-Bold", 11)
        return current_y

    def _add_summary_table(self, c, df, left_margin, current_y, width):
        # Use self.thresholds for all limit/alert values
        co2_limit = self._parse_threshold(self.thresholds['CO2']['recommended'])
        co2_alert = self._parse_threshold(self.thresholds['CO2']['alert'])
        nh3_limit = self._parse_threshold(self.thresholds['NH3']['recommended'])
        nh3_alert = self._parse_threshold(self.thresholds['NH3']['alert'])
        co2_mean = df['CO2'].mean()
        co2_max = df['CO2'].max()
        co2_above_limit = (df['CO2'] > co2_limit).sum()
        co2_above_alert = (df['CO2'] > co2_alert).sum()
        nh3_mean = df['NH3'].mean()
        nh3_max = df['NH3'].max()
        nh3_above_limit = (df['NH3'] > nh3_limit).sum()
        nh3_above_alert = (df['NH3'] > nh3_alert).sum()
        data = [
            ["Indicator", "Mean (ppm)", "Max (ppm)", "Count > Limit", "Count > Alert"],
            ["CO2", f"{co2_mean:.1f}", f"{co2_max:.1f}", f"{co2_above_limit}", f"{co2_above_alert}"],
            ["NH3", f"{nh3_mean:.1f}", f"{nh3_max:.1f}", f"{nh3_above_limit}", f"{nh3_above_alert}"],
        ]
        table_style = TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.Color(0.8, 1.0, 0.8)),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ])
        summary_table = Table(data, colWidths=[3*cm, 3*cm, 3*cm, 3*cm, 3*cm])
        summary_table.setStyle(table_style)
        table_width, table_height = summary_table.wrap(0, 0)
        x_position = (width - table_width) / 2
        summary_table.drawOn(c, x_position, current_y - table_height)
        return current_y - table_height - 20

    def _parse_threshold(self, value):
        # Extract numeric value from threshold string (e.g. '3000 ppm', '> 6000 ppm')
        import re
        match = re.search(r"([\d.]+)", value)
        if match:
            return float(match.group(1))
        return float('inf')

    def _plot_simple_evolution(self, df, variable, title):
        plt.figure(figsize=(8, 3))
        if not pd.api.types.is_datetime64_any_dtype(df['Date Time']):
            df['Date Time'] = pd.to_datetime(df['Date Time'])
        plt.plot(df['Date Time'], df[variable], color='green', linewidth=2)
        plt.title(title)
        plt.xlabel('Date Time')
        plt.ylabel(variable)
        plt.grid(True, linestyle='--', alpha=0.5)
        ax = plt.gca()
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45)
        plt.tight_layout()
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=150)
        plt.close()
        buf.seek(0)
        return buf

    def _plot_distribution(self, df, variable, title_hist, title_box):
        plt.figure(figsize=(12, 4))
        if variable not in df.columns:
            raise ValueError(f"Variable '{variable}' not found in dataframe")
        plt.subplot(1, 2, 1)
        plt.hist(df[variable].dropna(), bins=30, color='green', alpha=0.7)
        plt.title(title_hist)
        plt.xlabel(variable)
        plt.ylabel('Frequency')
        plt.subplot(1, 2, 2)
        plt.boxplot(df[variable].dropna(), orientation='vertical')
        plt.title(title_box)
        plt.ylabel(variable)
        plt.tight_layout()
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=150)
        plt.close()
        buf.seek(0)
        return buf

    def _add_diagnosis(self, c, df, current_y, width, left_margin):
        # Use self.thresholds for all limit/alert values
        co2_limit = self._parse_threshold(self.thresholds['CO2']['recommended'])
        co2_alert = self._parse_threshold(self.thresholds['CO2']['alert'])
        nh3_limit = self._parse_threshold(self.thresholds['NH3']['recommended'])
        nh3_alert = self._parse_threshold(self.thresholds['NH3']['alert'])
        total = len(df)
        co2_above_limit = (df['CO2'] > co2_limit).sum()
        co2_above_alert = (df['CO2'] > co2_alert).sum()
        nh3_above_limit = (df['NH3'] > nh3_limit).sum()
        nh3_above_alert = (df['NH3'] > nh3_alert).sum()
        co2_limit_pct = co2_above_limit / total * 100
        co2_alert_pct = co2_above_alert / total * 100
        nh3_limit_pct = nh3_above_limit / total * 100
        nh3_alert_pct = nh3_above_alert / total * 100
        messages = []
        if co2_above_alert > 0 or nh3_above_alert > 0:
            messages.append("Alert-level concentrations have been detected for CO2 and/or NH3. Immediate corrective measures are required to ensure sustainability.")
            apt = False
        elif co2_limit_pct > 5 or nh3_limit_pct > 5:
            messages.append("Several values for CO2 and/or NH3 exceed the recommended limits. Adjustments are needed before certification eligibility.")
            apt = False
        else:
            messages.append("All CO2 and NH3 values are within recommended thresholds. The farm complies with the environmental sustainability criteria.")
            apt = True
        conclusion = "Final Assessment: SUITABLE FOR CERTIFICATION" if apt else "Final Assessment: NOT SUITABLE FOR CERTIFICATION"
        messages.append(conclusion)
        style = ParagraphStyle(
            name="Diagnosis",
            fontName="Helvetica-Oblique",
            fontSize=9,
            leading=14,
            textColor=colors.green,
            alignment=TA_JUSTIFY,
        )
        paragraph = Paragraph("\n".join(messages), style)
        frame_width = width - left_margin * 2
        frame = Frame(left_margin, current_y - 100, frame_width, 100, showBoundary=0)
        frame.addFromList([paragraph], c)
        return current_y - 120
