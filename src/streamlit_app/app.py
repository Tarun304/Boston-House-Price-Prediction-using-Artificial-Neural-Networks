"""Streamlit UI for Boston House Price Prediction"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.streamlit_app.utils import FEATURE_INFO, LocalPredictor

# Page config
st.set_page_config(
    page_title="Boston House Price Predictor", page_icon="🏠", layout="wide"
)

# Custom CSS
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        color: white;
        box-shadow: 0 10px 20px rgba(0,0,0,0.2);
        margin: 20px 0;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 15px;
        font-size: 1.1rem;
    }
    </style>
""",
    unsafe_allow_html=True,
)


# Initialize predictor
@st.cache_resource
def get_predictor():
    predictor = LocalPredictor()
    success = predictor.load_artifacts()
    return predictor if success else None


# Header
st.markdown(
    '<p class="main-header">🏠 Boston House Price Predictor</p>', unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align: center; color: gray; margin-bottom: 2rem;'>Predict house prices using Machine Learning</p>",
    unsafe_allow_html=True,
)

# Load predictor
with st.spinner("🔄 Loading model..."):
    predictor = get_predictor()

if predictor is None:
    st.error("❌ Failed to load model. Please try again later.")
    st.stop()

st.markdown("---")

# Tabs
tab1, tab2 = st.tabs(["🎯 Predict Price", "📖 Feature Guide"])

# TAB 1: Prediction
with tab1:
    st.subheader("Enter House Features")

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        features = {}
        feature_list = list(FEATURE_INFO.keys())
        mid = len(feature_list) // 2

        # Left column
        with col1:
            for feature in feature_list[:mid]:
                info = FEATURE_INFO[feature]
                if feature == "CHAS":
                    features[feature] = st.selectbox(
                        f"{info['name']}",
                        options=[0, 1],
                        index=info["default"],
                        help=info["desc"],
                    )
                elif feature == "RAD":
                    features[feature] = st.slider(
                        f"{info['name']}",
                        min_value=int(info["range"][0]),
                        max_value=int(info["range"][1]),
                        value=info["default"],
                        help=info["desc"],
                    )
                else:
                    features[feature] = st.number_input(
                        f"{info['name']}",
                        min_value=float(info["range"][0]),
                        max_value=float(info["range"][1]),
                        value=float(info["default"]),
                        help=info["desc"],
                        format="%.4f",
                    )

        # Right column
        with col2:
            for feature in feature_list[mid:]:
                info = FEATURE_INFO[feature]
                if feature == "CHAS":
                    features[feature] = st.selectbox(
                        f"{info['name']}",
                        options=[0, 1],
                        index=info["default"],
                        help=info["desc"],
                    )
                elif feature == "RAD":
                    features[feature] = st.slider(
                        f"{info['name']}",
                        min_value=int(info["range"][0]),
                        max_value=int(info["range"][1]),
                        value=info["default"],
                        help=info["desc"],
                    )
                else:
                    features[feature] = st.number_input(
                        f"{info['name']}",
                        min_value=float(info["range"][0]),
                        max_value=float(info["range"][1]),
                        value=float(info["default"]),
                        help=info["desc"],
                        format="%.4f",
                    )

        submit_button = st.form_submit_button(
            "🔮 Predict Price", use_container_width=True
        )

    if submit_button:
        with st.spinner("Making prediction..."):
            try:
                prediction = predictor.predict(features)

                st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                st.markdown("### Predicted House Price")
                st.markdown(f"# ${prediction:,.2f}")
                st.markdown("</div>", unsafe_allow_html=True)

            except Exception as e:
                st.error(f"❌ Prediction failed: {str(e)}")

# TAB 2: Feature Guide
with tab2:
    st.subheader("📖 Feature Descriptions")
    st.write("Learn more about each input feature used by the model:")
    st.markdown("---")

    for feature, info in FEATURE_INFO.items():
        with st.expander(f"**{info['name']}** ({feature})"):
            st.write(f"**Description:** {info['desc']}")
            st.write(f"**Valid Range:** {info['range'][0]} - {info['range'][1]}")
            st.write(f"**Default Value:** {info['default']}")

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: gray;'>" "Built by Tarun Kumar Behera" "</p>",
    unsafe_allow_html=True,
)
