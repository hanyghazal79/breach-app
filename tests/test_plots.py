import matplotlib
import pandas as pd
from src.stats.plots import plot_km_by_group

def test_plot_km_by_group_returns_figure():
    df = pd.DataFrame({
        "followup_time_months": [5, 10, 15, 20, 25, 30],
        "event_recurrence": [1, 1, 0, 1, 0, 1],
        "risk_group": ["Low", "Low", "Mid", "Mid", "High", "High"],
    })
    fig = plot_km_by_group(df, "risk_group")
    assert isinstance(fig, matplotlib.figure.Figure)