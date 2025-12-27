"""Streamlit UI for Boston House Price Prediction"""

import streamlit as st
from utils import API_URL, FEATURE_INFO, check_api_health, predict_via_api

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

# Header
st.markdown(
    '<p class="main-header">🏠 Boston House Price Predictor</p>', unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align: center; color: gray; margin-bottom: 2rem;'>Predict house prices using Machine Learning</p>",
    unsafe_allow_html=True,
)

# Check API health
with st.spinner("🔄 Connecting to API..."):
    api_healthy = check_api_health()

if not api_healthy:
    st.error(
        f"❌ Cannot connect to API at {API_URL}. Please make sure FastAPI is running!"
    )
    st.info("💡 Run FastAPI with: `uvicorn src.api.app:app --reload`")
    st.stop()
else:
    st.success(f"✅ Connected to API at {API_URL}")

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
                prediction = predict_via_api(features)

                st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                st.markdown("### Predicted House Price")
                st.markdown(f"# ${prediction:,.2f}")
                st.markdown("</div>", unsafe_allow_html=True)

            except Exception as e:
                st.error(f"❌ {str(e)}")

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
