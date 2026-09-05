"""
Plotting helpers for the Streamlit UI. Not meaningfully testable for
"correctness" the way score/stats functions are -- the useful test here
is just "does this run and return a Figure", not an exact pixel match.
"""

import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter


def plot_km_by_group(df, group_col, time_col="followup_time_months",
                      event_col="event_recurrence"):
    fig, ax = plt.subplots(figsize=(7, 5))
    for group_name, group_df in df.groupby(group_col):
        kmf = KaplanMeierFitter()
        kmf.fit(group_df[time_col], group_df[event_col],
                label=f"{group_name} (n={len(group_df)})")
        kmf.plot_survival_function(ax=ax)
    ax.set_xlabel("Follow-up (months)")
    ax.set_ylabel("Recurrence-free survival probability")
    ax.set_title("Kaplan-Meier by risk group")
    return fig