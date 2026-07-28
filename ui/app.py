import os
import time

import pandas as pd
import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")
REPORTS_DIR = os.path.join(os.path.dirname(__file__), "..", "reports")

CLASS_NAMES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

st.set_page_config(page_title="Student Engagement Risk", layout="wide")
st.title("Student Engagement Risk Detector")
st.caption("Facial-affect early warning signal, built on top of my earlier tabular dropout-risk project")

tab_predict, tab_viz, tab_upload, tab_status = st.tabs(
    ["Predict", "Visualizations", "Upload & Retrain", "Status"]
)

with tab_predict:
    st.subheader("Predict from a single face image")
    uploaded = st.file_uploader("upload a face photo", type=["png", "jpg", "jpeg"])
    if uploaded is not None:
        col1, col2 = st.columns([1, 2])
        col1.image(uploaded, width=220)
        if col2.button("Predict"):
            files = {"file": (uploaded.name, uploaded.getvalue())}
            try:
                r = requests.post(f"{API_URL}/predict", files=files, timeout=30)
                r.raise_for_status()
                result = r.json()
                col2.success(f"Predicted: **{result['label']}** ({result['confidence'] * 100:.1f}% confidence)")
                probs_df = pd.DataFrame(
                    list(result["probs"].items()), columns=["class", "probability"]
                ).sort_values("probability", ascending=False)
                col2.bar_chart(probs_df.set_index("class"))
            except Exception as e:
                col2.error(f"prediction failed: {e}")

with tab_viz:
    st.subheader("Dataset insights")
    try:
        stats = requests.get(f"{API_URL}/stats", timeout=10).json()
        counts_df = pd.DataFrame(
            list(stats["train_counts"].items()), columns=["class", "count"]
        ).sort_values("count", ascending=False)
        st.bar_chart(counts_df.set_index("class"))
    except Exception as e:
        st.error(f"could not reach API for live stats: {e}")

    st.markdown("**What the data is telling us**")
    st.markdown(
        "1. *Class imbalance* - disgust has only a fraction of the images that happy or "
        "neutral have. Same issue as the Enrolled class in my earlier tabular project, so "
        "I'm reporting macro-averaged metrics here too rather than plain accuracy.\n\n"
        "2. *Brightness by class* - happy and surprise faces are on average brighter than "
        "sad and fear ones, mostly because of open mouths/eyes and more visible teeth "
        "reflecting light. It's a weak but real signal, see the notebook for the numbers.\n\n"
        "3. *Confusion pattern* - the model mixes up fear and sad far more than it mixes up "
        "happy and angry. Those two are close together in facial muscle activation (brow "
        "and mouth), which lines up with the confusion matrix in the notebook."
    )

    conf_matrix_path = os.path.join(REPORTS_DIR, "confusion_matrix.png")
    brightness_path = os.path.join(REPORTS_DIR, "brightness_by_class.png")
    cols = st.columns(2)
    if os.path.exists(conf_matrix_path):
        cols[0].image(conf_matrix_path, caption="Confusion matrix (test set)")
    if os.path.exists(brightness_path):
        cols[1].image(brightness_path, caption="Mean brightness by class")

with tab_upload:
    st.subheader("Upload new labeled images")
    label = st.selectbox("label for these images", CLASS_NAMES)
    uploaded_files = st.file_uploader(
        "choose one or more images", type=["png", "jpg", "jpeg"], accept_multiple_files=True
    )
    if st.button("Upload batch"):
        if not uploaded_files:
            st.warning("pick at least one file first")
        else:
            files = [("files", (f.name, f.getvalue())) for f in uploaded_files]
            try:
                r = requests.post(f"{API_URL}/upload", data={"label": label}, files=files, timeout=60)
                r.raise_for_status()
                st.success(f"saved {r.json()['saved']} images under '{label}'")
            except Exception as e:
                st.error(f"upload failed: {e}")

    st.divider()
    st.subheader("Trigger retraining")
    st.caption("fine-tunes the current saved model on whatever's been uploaded since the last run")
    if st.button("Retrain now"):
        with st.spinner("fine-tuning on new data, this can take a minute..."):
            try:
                r = requests.post(f"{API_URL}/retrain", timeout=900)
                r.raise_for_status()
                result = r.json()
            except Exception as e:
                st.error(f"retrain call failed: {e}")
                result = None
        if result and result["status"] == "ok":
            st.success(
                f"done - macro F1 went from {result['old_macro_f1']:.3f} to "
                f"{result['new_macro_f1']:.3f} using {result['n_new_images']} new images"
            )
        elif result:
            st.info("no new uploaded images to retrain on yet")

with tab_status:
    st.subheader("Service status")
    try:
        health = requests.get(f"{API_URL}/health", timeout=10).json()
        col1, col2 = st.columns(2)
        col1.metric("uptime (s)", health["uptime_seconds"])
        last_retrain = health["last_retrain"]
        if last_retrain:
            when = time.strftime("%Y-%m-%d %H:%M", time.localtime(last_retrain["ran_at"]))
            col2.metric("last retrain", when)
        else:
            col2.metric("last retrain", "never")
    except Exception as e:
        st.error(f"API unreachable: {e}")

    try:
        stats = requests.get(f"{API_URL}/stats", timeout=10).json()
        recent = pd.DataFrame(stats["recent_requests"])
        if not recent.empty:
            recent["time"] = pd.to_datetime(recent["ts"], unit="s")
            st.line_chart(recent.set_index("time")["latency_ms"])
        else:
            st.info("no prediction requests logged yet - try the Predict tab")
    except Exception as e:
        st.error(str(e))
