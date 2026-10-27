import json

import pandas as pd
import streamlit as st

from predict_utils import ROOT, load_model, read_lots, score_lots

st.set_page_config(page_title="SECOM Yield Risk Scoring", layout="wide")
st.title("Semiconductor lot risk scoring (SECOM)")
st.caption("Portfolio project by Akash Dathaprasad. Model trained on real UCI SECOM fab data.")


@st.cache_resource
def get_model():
    return load_model()


tab_score, tab_results = st.tabs(["Score lots", "Project results"])

with tab_score:
    st.write("Upload sensor readings for production lots. The model gives each lot a failure risk score, "
             "so engineers can inspect the riskiest lots first. Accepted files: the raw UCI `secom.data` file "
             "(space separated, 590 columns, no header) or a CSV with a header `s001 ... s590`.")
    st.warning("Honest note: the model is only modestly better than random (test ROC-AUC about 0.70 on 236 future lots). "
               "Use it to prioritise inspections, not to make final pass/fail decisions.")
    top_pct = st.slider("Inspect the top X% highest-risk lots", 5, 100, 20, step=5)
    up = st.file_uploader("Upload sensor file", type=["data", "csv", "txt"])
    if up is not None:
        try:
            lots = read_lots(up)
            scored = score_lots(get_model(), lots)
            n_flag = max(1, round(len(scored) * top_pct / 100))
            scored["inspect"] = scored["risk_rank"] <= n_flag
            c1, c2, c3 = st.columns(3)
            c1.metric("Lots scored", len(scored))
            c2.metric("Lots to inspect", n_flag)
            c3.metric("Highest risk score", f"{scored['risk_score'].max():.2f}")
            st.dataframe(scored, width="stretch", height=400)
            st.bar_chart(scored.set_index("risk_rank")["risk_score"])
            st.download_button("Download scores (CSV)", scored.to_csv(index=False), "lot_risk_scores.csv", "text/csv")
        except Exception as e:
            st.error(f"Could not process the file: {e}")
    else:
        st.info("No file yet. Get the data from https://archive.ics.uci.edu/dataset/179/secom (file `secom.data`).")

with tab_results:
    summary = json.loads((ROOT / "results" / "summary.json").read_text())
    tm = summary["test_metrics"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Test ROC-AUC", tm["roc_auc"], help=f"95% CI {tm['roc_auc_95ci']}")
    c2.metric("Test PR-AUC", tm["pr_auc"], help=f"Random guess = {tm['random_guess_pr_auc']}")
    c3.metric("Test lots / failures", f"{tm['test_lots']} / {tm['test_failures']}")
    st.subheader("Model comparison (time-series cross-validation)")
    st.dataframe(pd.DataFrame(summary["cv_table"]), width="stretch")
    st.subheader("Inspect only the top-risk lots (test set)")
    st.dataframe(pd.DataFrame(summary["gains_table"]), width="stretch")
    st.subheader("Cost analysis (test set)")
    st.dataframe(pd.DataFrame(summary["cost_table"]), width="stretch")
    for fig in sorted((ROOT / "results" / "figures").glob("*.png")):
        st.image(str(fig), width="stretch")
