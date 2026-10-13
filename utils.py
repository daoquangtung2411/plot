"""Data loading helpers."""

import io

import numpy as np
import pandas as pd
import streamlit as st


@st.cache_data
def load_file(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    raw = uploaded_file.getvalue()
    if name.endswith(".csv"):
        return pd.read_csv(io.BytesIO(raw))
    if name.endswith(".tsv") or name.endswith(".txt"):
        return pd.read_csv(io.BytesIO(raw), sep="\t")
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(io.BytesIO(raw))
    raise ValueError(f"Unsupported file type: {uploaded_file.name}")


@st.cache_data
def make_sample_data(seed: int = 0) -> pd.DataFrame:
    """Synthetic lab-style dataset: groups x timepoints x replicates."""
    rng = np.random.default_rng(seed)
    groups = ["Control", "Treatment A", "Treatment B"]
    timepoints = [0, 6, 12, 24, 48]
    n_replicates = 8

    rows = []
    group_effect = {"Control": 0.0, "Treatment A": 1.8, "Treatment B": -1.2}
    for group in groups:
        for t in timepoints:
            for rep in range(n_replicates):
                baseline = 10 + 0.15 * t
                effect = group_effect[group] * np.log1p(t)
                noise = rng.normal(0, 1.2)
                value = baseline + effect + noise
                covariate = value * 0.8 + rng.normal(0, 1.5)
                rows.append(
                    {
                        "sample_id": f"{group[:4]}_{t}h_r{rep}",
                        "group": group,
                        "time_h": t,
                        "replicate": rep,
                        "expression": round(value, 3),
                        "viability_pct": round(np.clip(95 - 0.3 * t + rng.normal(0, 3), 40, 100), 2),
                        "covariate_marker": round(covariate, 3),
                    }
                )
    return pd.DataFrame(rows)
