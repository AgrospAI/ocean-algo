# CSV to PDF Report Generator

A powerful Python algorithm that automatically generates comprehensive PDF reports from CSV data files. The algorithm performs extensive data analysis, creates visualizations, and produces a professional PDF report with multiple sections.

## Features

### 1. Data Analysis
- Automatic datetime column detection
- Numeric column analysis
- Data quality assessment
- Missing data detection
- Time series analysis
- Statistical summaries

### 2. Visualizations
- Histograms for each numeric column
- Time series plots with moving averages
- Correlation matrices and scatter plots
- Diurnal pattern analysis
- 24-hour forecasts

### 3. PDF Report Sections
- Cover page with logo support
- Executive summary
- Data quality assessment
- Summary statistics
- Histograms
- Time series trends
- Correlation analysis
- Diurnal patterns
- 24-hour forecasts
- Anomaly detection
- Conclusions and recommendations
- Raw data preview

## Requirements

- Python 3.x
- uv package manager (recommended) or pip
- Required packages:
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - reportlab
  - oceanprotocol_job_details

## Installation

### Using uv (Recommended)

1. Install uv:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Create and activate a virtual environment:
```bash
uv venv
```

3. Install dependencies:
```bash
uv pip install -r requirements.txt
```

### Using pip (Alternative)

1. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Unix/macOS
# or
.venv\Scripts\activate  # On Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Input Data Format

The algorithm expects a CSV file with:
- At least one column that can be converted to datetime
- One or more numeric columns

Example CSV format:
```csv
timestamp,temperature,humidity
2021-01-01 00:00:00,25.5,65.2
2021-01-01 01:00:00,25.3,66.1
...
```

## Configuration Parameters
The algorithm includes several configurable parameters, which can be set in the `algoCustomData.json` file located in the `_data/inputs/` directory:

```json
{
  "anomaly_multiplier": 4.0,
  "forecast_roll_window": 6,
  "default_temp_threshold": [5.0, 30.0],
  "default_hum_threshold": [30.0, 90.0],
  "min_points_for_forecast": 24
}
```

To customize these parameters, edit the values in the JSON file as needed.

## Usage

### Local Development

1. Place your CSV data file in the `_data/inputs/` directory
2. Run the main script:
```bash
python src/main.py
```

### Docker Deployment

1. Build the Docker image:
```bash
docker build -t csv-pdf-report .
```

2. Run the container:
```bash
docker run -v /path/to/data:/data csv-pdf-report
```

## Output

The algorithm generates:
1. A comprehensive PDF report in the `temp_report.pdf` directory
2. Individual visualization files for each analysis
3. Statistical summaries and anomaly reports

## Project Structure

```
csv-pdf-report/
├── algorithm/
│   ├── src/
│   │   ├── implementation/
│   │   │   └── algorithm.py    # Main algorithm implementation
│   │   └── main.py            # Entry point
│   └── Dockerfile
├── _data/
│   └── inputs/                # Input CSV files
├── requirements.txt           # Python dependencies
└── README.md
```

## Customization

### Adding a Logo
Place a `logo.png` file in the same directory as your input CSV file to include it in the report's cover page.

### Modifying Report Style
The report's visual style can be customized by modifying the `_get_report_styles` method in `algorithm.py`. This includes:
- Font families and sizes
- Colors
- Spacing
- Layout

### Adjusting Analysis Parameters
Key analysis parameters can be modified at the top of `algorithm.py`:
- Anomaly detection thresholds
- Forecast window sizes
- Default value ranges
- Minimum data points required

## Error Handling

The algorithm includes comprehensive error handling for:
- Missing or invalid input files
- Data parsing errors
- Visualization generation failures
- PDF generation issues
