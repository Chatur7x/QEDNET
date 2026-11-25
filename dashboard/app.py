"""Legacy Streamlit landing page; the Vite frontend is the active dashboard."""

import streamlit as st

st.set_page_config(
    page_title="QED-Net Research Platform",
    page_icon="🔬",
    layout="wide",
)

st.title("QED-Net Research Platform")
st.warning(
    "This legacy Streamlit page does not train models, load predictions, or show "
    "measured results. Its former demonstration outputs were synthetic and have "
    "been removed."
)
st.markdown(
    "Use the API-connected Vite dashboard at **http://localhost:5173**. "
    "Start the FastAPI backend with `python -m api.main` and the frontend with "
    "`npm run dev` from `frontend/`."
)
st.caption(
    "QED-Net is for research and education. It is not a medical device, is not "
    "clinically validated, and must not be used for diagnosis or real patient care."
)
