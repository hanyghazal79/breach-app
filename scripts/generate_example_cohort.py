"""
Generates the SYNTHETIC example cohort used by the "Load example dataset"
button (data/example_cohort.csv). This is demonstration data only -- the
correlation structure is a simplified simulation, not derived from real
BREACH biology or any real patient population. Any discrimination result
shown on this dataset demonstrates the STATISTICAL MACHINERY working
correctly, not evidence about BREACH's real-world performance.
"""

import numpy as np
import pandas as pd


def generate_example_cohort(n: int = 60, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    age = rng.normal(58, 10, n).clip(30, 85)
    menopausal_status = np.where(age >= 50, "post", "pre")
    grade = rng.choice([1, 2, 3], size=n, p=[0.25, 0.45, 0.30])
    size = rng.gamma(shape=3, scale=8, size=n).clip(3, 90)
    nodes = rng.poisson(1.2, n)
    lvi = rng.choice(["absent", "present"], size=n, p=[0.75, 0.25])
    ki67 = rng.normal(25, 15, n).clip(1, 95)
    er_status = rng.choice(["positive", "negative"], size=n, p=[0.8, 0.2])
    neutrophils = rng.normal(4.5, 1.5, n).clip(1, 12)
    lymphocytes = rng.normal(1.9, 0.6, n).clip(0.3, 4.5)
    albumin = rng.normal(40, 5, n).clip(25, 50)
    hemoglobin = rng.normal(12.5, 1.3, n).clip(8, 16)
    endocrine_5y = rng.choice([True, False], size=n, p=[0.6, 0.4])
    dfs_5y = endocrine_5y & (rng.random(n) < 0.85)

    # Latent risk combines established factors (size, grade, nodes) AND
    # BREACH's additional inputs (Ki67, NLR, LVI, albumin, Hgb) plus noise --
    # not engineered to favor any single score over another.
    nlr = neutrophils / lymphocytes
    latent_risk = (
        0.03 * size + 0.8 * grade + 0.5 * nodes
        + 0.04 * ki67 + 0.3 * nlr + 1.0 * (lvi == "present")
        - 0.05 * albumin - 0.1 * hemoglobin
        + rng.normal(0, 3, n)
    )
    risk_z = (latent_risk - latent_risk.mean()) / latent_risk.std()

    event_prob = 1 / (1 + np.exp(-(risk_z - 0.2)))
    event_recurrence = (rng.random(n) < event_prob).astype(int)
    followup_time_months = (90 - 15 * risk_z + rng.normal(0, 8, n)).clip(2, 120)

    return pd.DataFrame({
        "patient_id": [f"DEMO-{i:03d}" for i in range(n)],
        "age_years": age.round(0), "menopausal_status": menopausal_status,
        "tumor_size_mm": size.round(0), "tumor_grade": grade,
        "nodes_positive_count": nodes, "lvi_status": lvi,
        "ki67_percent": ki67.round(1), "er_status": er_status,
        "neutrophils_10e9_L": neutrophils.round(2), "lymphocytes_10e9_L": lymphocytes.round(2),
        "albumin_g_L": albumin.round(1), "hemoglobin_g_dL": hemoglobin.round(1),
        "endocrine_therapy_completed_5y": endocrine_5y, "disease_free_at_5y": dfs_5y,
        "followup_time_months": followup_time_months.round(0), "event_recurrence": event_recurrence,
    })


if __name__ == "__main__":
    df = generate_example_cohort()
    df.to_csv("data/example_cohort.csv", index=False)
    print(f"Wrote {len(df)} synthetic patients to data/example_cohort.csv")