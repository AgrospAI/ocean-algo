import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta

class TrendForecaster:
    """
    Provides linear and rolling-average forecasts for temperature and humidity.
    """
    def __init__(self, df, time_col="entity_ts", temp_col="temperature", hum_col="humidity"):
        self.df = df.copy()
        self.time_col = time_col
        self.temp_col = temp_col
        self.hum_col = hum_col
        self.df = self.df.sort_values(self.time_col)

    def linear_forecast(self, forecast_hours=24, window_days=3):
        # Use last N days for trend, or all if short
        recent = self.df[self.df[self.time_col] > (self.df[self.time_col].max() - pd.Timedelta(days=window_days))]
        x = (recent[self.time_col] - recent[self.time_col].min()).dt.total_seconds() / 3600  # hours
        results = {}
        for col in [self.temp_col, self.hum_col]:
            y = recent[col]
            if len(x) < 2:
                results[col] = None
                continue
            coeffs = np.polyfit(x, y, 1)  # Linear fit
            future_hours = np.arange(x.max(), x.max() + forecast_hours + 1)
            future_times = [recent[self.time_col].min() + pd.Timedelta(hours=float(h)) for h in future_hours]
            forecast_vals = np.polyval(coeffs, future_hours)
            results[col] = {
                "future_times": future_times,
                "forecast_vals": forecast_vals,
                "slope": coeffs[0],
                "intercept": coeffs[1],
                "last_val": y.iloc[-1] if len(y) > 0 else None,
                "forecast_end": forecast_vals[-1] if len(forecast_vals) > 0 else None
            }
        return results

    def plot_forecast(self, out_path):
        results = self.linear_forecast()
        fig, axs = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
        for i, (col, label, color) in enumerate([
            (self.temp_col, "Temperature (°C)", "tab:red"),
            (self.hum_col, "Humidity (%)", "tab:blue")]):
            axs[i].plot(self.df[self.time_col], self.df[col], label="Observed", color=color, alpha=0.7)
            axs[i].plot(self.df[self.time_col], self.df[col].rolling(window=12).mean(), label="1h-MA", color=color, linestyle="--", alpha=0.7)
            if results[col] is not None:
                axs[i].plot(results[col]["future_times"], results[col]["forecast_vals"], label="Linear Forecast (next 24h)", color=color, linestyle=":", alpha=0.9)
                axs[i].axvline(self.df[self.time_col].max(), color="#888888", linestyle="--", lw=1)
            axs[i].set_ylabel(label)
            axs[i].legend(fontsize=8, loc="best")
        axs[1].set_xlabel("Timestamp")
        fig.autofmt_xdate(rotation=30, ha="right")
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.25)
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return results

    def forecast_summary(self, results):
        lines = []
        for col, label, unit in [
            (self.temp_col, "Temperature", "°C"),
            (self.hum_col, "Humidity", "%")]:
            r = results.get(col)
            if r is None:
                lines.append(f"{label}: Not enough data for forecast.")
                continue
            delta = r["forecast_end"] - r["last_val"] if r["last_val"] is not None and r["forecast_end"] is not None else None
            if delta is not None:
                lines.append(f"{label} is forecast to change by {delta:+.2f}{unit} over the next 24 hours (linear trend).")
            else:
                lines.append(f"{label}: Forecast unavailable.")
        return "<br/>".join(lines)
