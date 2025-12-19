# streamlit_dashboard_updated.py
import streamlit as st
import pandas as pd
import plotly.express as px
import os
import glob
from pathlib import Path

BASE = Path(r"D:\Codes\LabProj\Telegram_New\Result")

st.title("Lab Task: MohammadMahdi Shojaey — Telegram results")
st.sidebar.header("Navigation")

menu = [
    "Classification Reports For SAE",
    "Classification Reports For CNN",
    "CNN Prefix Scan",
    "CNN Comparison (Permute vs Prefix)"
]
choice = st.sidebar.radio("Go to", menu)

# helper to read CSV safely
def read_csv_safe(p):
    try:
        return pd.read_csv(p)
    except Exception as e:
        st.error(f"Failed to read {p}: {e}")
        return None

# -------------------------------------------------------------------------
# SAE reports (folder: SAE)
# -------------------------------------------------------------------------
if choice == "Classification Reports For SAE":
    sae_dir = BASE / "SAE"
    if not sae_dir.exists():
        st.error(f"SAE result folder not found: {sae_dir}")
        st.stop()

    # look for files with names like report_train1_test2.csv or sae_report_...
    patterns = ["report_train*_test*.csv", "sae_report_train*_test*.csv", "report_train*_test*.csv"]
    csvs = []
    for pat in patterns:
        csvs += sorted(sae_dir.glob(pat))
    csvs = sorted(set(csvs))

    if not csvs:
        st.error(f"No SAE CSV reports found in: {sae_dir}")
        st.stop()

    for p in csvs:
        df = read_csv_safe(p)
        if df is None:
            continue
        fname = p.stem
        # Try to parse train/test from filename
        fn = fname.replace("sae_", "").replace("report_", "").replace("sae_report_", "")
        parts = fn.split("_")
        # Expect patterns like train1_test2 or train1_test2_more
        train = next((s.replace("train", "") for s in parts if s.startswith("train")), "?")
        test = next((s.replace("test", "") for s in parts if s.startswith("test")), "?")

        st.subheader(f"SAE Report — Train {train} → Test {test}")
        st.dataframe(df)
        if "overall" in df["Class"].values:
            overall_acc = df.loc[df["Class"] == "overall", "Accuracy"].values[0]
            st.metric("Overall Accuracy", f"{overall_acc:.4f}")

    st.stop()

# -------------------------------------------------------------------------
# CNN classification reports (prefer ReducedCNN, fallback to CNN)
# -------------------------------------------------------------------------
elif choice == "Classification Reports For CNN":
    # Check ReducedCNN then CNN
    cnn_dirs = [BASE / "ReducedCNN", BASE / "CNN"]
    csvs = []
    for d in cnn_dirs:
        if d.exists():
            csvs += sorted(d.glob("cnn_report_train*_test*.csv"))
            csvs += sorted(d.glob("cnn_train*_test*.csv"))  # alternate naming
    csvs = sorted(set(csvs))

    if not csvs:
        st.error("No CNN report CSV files found in ReducedCNN or CNN folders.")
        st.stop()

    for p in csvs:
        df = read_csv_safe(p)
        if df is None:
            continue
        fname = p.stem
        fn = fname.replace("cnn_", "").replace("cnn_report_", "")
        parts = fn.split("_")
        train = next((s.replace("train", "") for s in parts if s.startswith("train")), None)
        test  = next((s.replace("test", "")  for s in parts if s.startswith("test")), None)
        # fallback parsing for names like cnn_train1_test2 or cnn_train1_test2_prefixX
        if train is None or test is None:
            # try tokens
            tokens = [t for t in parts if t.startswith("train") or t.startswith("test")]
            train = train or (tokens[0].replace("train", "") if tokens else "?")
            test  = test  or (tokens[1].replace("test", "") if len(tokens) > 1 else "?")

        st.subheader(f"CNN Report — Train {train} → Test {test}")
        st.dataframe(df)
        if "overall" in df["Class"].values:
            overall_acc = df.loc[df["Class"] == "overall", "Accuracy"].values[0]
            st.metric("Overall Accuracy", f"{overall_acc:.4f}")

    st.stop()

# -------------------------------------------------------------------------
# CNN Prefix Scan — look in ReducedCNN (preferred) then CNN
# -------------------------------------------------------------------------
elif choice == "CNN Prefix Scan":
    candidates = [BASE / "ReducedCNN" / "cnn_prefix_scan_report.csv",
                  BASE / "CNN" / "cnn_prefix_scan_report.csv"]
    csv_file = next((c for c in candidates if c.exists()), None)
    if csv_file is None:
        st.error("cnn_prefix_scan_report.csv not found in ReducedCNN or CNN folders.")
        st.stop()

    df = read_csv_safe(csv_file)
    if df is None:
        st.stop()

    st.subheader("CNN Prefix Scan Results")
    st.dataframe(df)
    fig_acc = px.line(df, x="PrefixBytes", y="Accuracy", markers=True, title="Accuracy vs PrefixBytes")
    fig_acc.update_layout(xaxis=dict(autorange=True))
    st.plotly_chart(fig_acc, use_container_width=True)
    fig_f1 = px.line(df, x="PrefixBytes", y="F1", markers=True, title="F1 vs PrefixBytes")
    fig_f1.update_layout(xaxis=dict(autorange=True))
    st.plotly_chart(fig_f1, use_container_width=True)
    st.stop()

# -------------------------------------------------------------------------
# CNN Comparison (Permutation vs Prefix). Permutation usually lives in
# FirstFourPermutation or in ReducedCNN as cnn_permute4_report.csv
# -------------------------------------------------------------------------
elif choice == "CNN Comparison (Permute vs Prefix)":
    perm_candidates = [
        BASE / "FirstFourPermutation" / "cnn_first4perm_report.csv",
        BASE / "FirstFourPermutation" / "cnn_permute4_report.csv",
        BASE / "ReducedCNN" / "cnn_permute4_report.csv",
        BASE / "CNN" / "cnn_permute4_report.csv"
    ]
    perm_file = next((p for p in perm_candidates if p.exists()), None)

    prefix_candidates = [
        BASE / "ReducedCNN" / "cnn_prefix_scan_report.csv",
        BASE / "CNN" / "cnn_prefix_scan_report.csv"
    ]
    prefix_file = next((p for p in prefix_candidates if p.exists()), None)

    if perm_file is None:
        st.error("Permutation report not found in FirstFourPermutation/ReducedCNN/CNN.")
        st.stop()
    if prefix_file is None:
        st.error("Prefix-scan report not found in ReducedCNN or CNN.")
        st.stop()

    df_perm = read_csv_safe(perm_file)
    df_pref = read_csv_safe(prefix_file)
    if df_perm is None or df_pref is None:
        st.stop()

    st.subheader("Permutation vs Prefix Scan")
    st.markdown("Permutation (First-4 bytes neutralization/randomization)")
    st.dataframe(df_perm)
    st.markdown("Prefix Scan")
    st.dataframe(df_pref)

    df_perm = df_perm.assign(Source="Permute4")
    df_pref = df_pref.assign(Source="PrefixScan")

    df_comb = pd.concat([df_perm, df_pref], ignore_index=True)
    df_comb = df_comb.sort_values(["Source", "PrefixBytes"], ascending=[True, False])

    fig_cmp_acc = px.line(df_comb, x="PrefixBytes", y="Accuracy", color="Source",
                          markers=True, title="Accuracy Comparison")
    fig_cmp_acc.update_layout(xaxis=dict(autorange=True))
    st.plotly_chart(fig_cmp_acc, use_container_width=True)

    fig_cmp_f1 = px.line(df_comb, x="PrefixBytes", y="F1", color="Source",
                         markers=True, title="F1 Comparison")
    fig_cmp_f1.update_layout(xaxis=dict(autorange=True))
    st.plotly_chart(fig_cmp_f1, use_container_width=True)

    st.stop()

# fallback
st.info("Select a view from the sidebar.")
