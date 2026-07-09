"""
Quick demo UI so I can actually show this off without someone needing to
hit the API with curl. Run with:

    streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

import streamlit as st
from PIL import Image

# so this works when run directly with `streamlit run` from repo root
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.inference.predictor import DefectPredictor
from src.utils.config import load_config, get_device

st.set_page_config(page_title="Defect Detection Demo", layout="wide")

st.title("Industrial Defect Detection")
st.caption("Upload a product image and the model will flag whether it looks defective, "
           "plus show roughly where it thinks the problem is.")


@st.cache_resource
def get_predictor(mode):
    cfg = load_config("configs/config.yaml")
    device = get_device(cfg.project.device)
    return DefectPredictor(cfg, mode=mode, device=device)


with st.sidebar:
    st.header("Settings")
    mode = st.selectbox("Model", ["patchcore", "autoencoder", "classifier"], index=0,
                         help="patchcore = feature-embedding + nearest neighbor (best result, "
                              "0.994 auroc on real data, see README), "
                              "autoencoder = reconstruction error (weaker, 0.566 auroc - "
                              "left in here to show the comparison), "
                              "classifier = supervised good/defective head")
    st.markdown("---")
    st.markdown(
        "No checkpoint trained yet? The predictor still runs on the "
        "randomly-initialized model so you can click around, the scores "
        "just won't mean anything until you train it."
    )

uploaded = st.file_uploader("Upload a product image", type=["png", "jpg", "jpeg"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    predictor = get_predictor(mode)

    with st.spinner("running inference..."):
        result = predictor.predict(image, return_heatmap=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Input")
        st.image(image, use_column_width=True)
    with col2:
        st.subheader("Anomaly Heatmap")
        if "heatmap_overlay" in result:
            st.image(result["heatmap_overlay"], use_column_width=True)
        else:
            st.info("no heatmap for this mode")

    st.markdown("---")
    verdict = "DEFECTIVE" if result["is_defective"] else "OK"
    color = "red" if result["is_defective"] else "green"
    st.markdown(f"### Verdict: :{color}[{verdict}]")
    st.metric("Anomaly score", f"{result['score']:.4f}")
else:
    st.info("upload an image to get started, or grab a sample from data/mvtec/<category>/test/")
