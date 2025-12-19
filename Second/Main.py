import streamlit as st
import pandas as pd
import plotly.express as px
import os
import glob

st.title("Lab Task: MohammadMahdi Shojaey")
st.sidebar.header("Navigation")
menu = [
    "Classification Reports For SAE",
    "Classification Reports For CNN",
    "CNN Prefix Scan (Telegram)"
]
choice = st.sidebar.radio("Go to", menu)

# --------------------------------------------------------------------------- #
# CORRECT TELEGRAM RESULT PATHS
# --------------------------------------------------------------------------- #
if choice == "Classification Reports For SAE":
    data_path = r"D:\Codes\LabProj\Telegram_New\Result\SAE"
    pattern = "sae_report_train*_test*.csv"

elif choice == "Classification Reports For CNN":
    data_path = r"D:\Codes\LabProj\Telegram_New\Result\CNN"
    pattern = "cnn_report_train*_test*.csv"

elif choice == "CNN Prefix Scan (Telegram)":
    csv_file = r"D:\Codes\LabProj\Telegram_New\Result\ReducedCNN\cnn_prefix_scan_report.csv"
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        st.error(f"File not found: {csv_file}")
        st.stop()

    st.subheader("CNN Prefix Scan Results (Telegram Dataset)")
    st.dataframe(df)

    # Accuracy vs PrefixBytes
    fig_acc = px.line(
        df,
        x="PrefixBytes",
        y="Accuracy",
        markers=True,
        title="Accuracy vs. PrefixBytes (Telegram)"
    )
    fig_acc.update_layout(xaxis=dict(autorange='reversed'))
    st.plotly_chart(fig_acc, use_container_width=True)

    # F1 vs PrefixBytes
    fig_f1 = px.line(
        df,
        x="PrefixBytes",
        y="F1",
        markers=True,
        title="F1 Score vs. PrefixBytes (Telegram)"
    )
    fig_f1.update_layout(xaxis=dict(autorange='reversed'))
    st.plotly_chart(fig_f1, use_container_width=True)

    st.markdown("---")
    st.stop()

# --------------------------------------------------------------------------- #
# EXISTING CLASSIFICATION REPORT LOGIC
# --------------------------------------------------------------------------- #
csv_files = glob.glob(os.path.join(data_path, pattern))
if not csv_files:
    st.error(f"No report CSV files found in {data_path}")

for file in sorted(csv_files):
    df = pd.read_csv(file)
    filename = os.path.basename(file).replace(".csv", "")
    base = filename.replace("report_", "").replace("cnn_", "")
    parts = base.split("_")
    train = parts[0].replace("train", "")
    test = parts[-1].replace("test", "")
    st.subheader(f"Classification Report: Train {train} → Test {test}")

    df_disp = df.drop(columns=["Accuracy"], errors="ignore")
    st.dataframe(df_disp)

    if "overall" in df["Class"].values:
        overall_acc = df.loc[df["Class"] == "overall", "Accuracy"].values[0]
        st.metric("Overall Accuracy", f"{overall_acc:.4f}")

    # Show confusion matrix for Telegram class
    df_telegram = df[df['Class'].str.lower() == 'telegram']
    for _, row in df_telegram.iterrows():
        cm = [[row['TP'], row['FN']], [row['FP'], row['TN']]]
        fig_cm = px.imshow(
            cm,
            text_auto=True,
            aspect="auto",
            labels={"x": "Predicted", "y": "Actual"},
            x=["Predicted Positive", "Predicted Negative"],
            y=["Actual Positive", "Actual Negative"],
            title=f"Confusion Matrix for {row['Class']}"
        )
        st.plotly_chart(fig_cm, use_container_width=True)

st.markdown("---")
