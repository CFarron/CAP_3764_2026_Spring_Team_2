import streamlit as st
import requests
import pandas as pd


# Run the app: 
    # 1-Open another terminal
    # 2-Activate env (run in terminal: conda activate fastapi_streamlit)
    # 3-Set your working directory to the folder where you have app.py (run in terminal: cd your_path/)
    # 4-Execute app.py (run in terminal: streamlit run app.py)



API_BASE = "http://localhost:8000"

st.title("FastAPI + Streamlit + Batch Demo")

tab_predict, tab_batch = st.tabs(
    ["🔮 Predict", "📂 Batch Predict"]
)

# -------------------------------------------------
# Tab 1: Predict
# -------------------------------------------------
with tab_predict:
    st.subheader("Single Prediction")

    col1, col2 = st.columns(2)

    with col1:
        mean_radius = st.number_input("mean_radius", 5.0, 30.0, 14.0)
        mean_texture = st.number_input("mean_texture", 5.0, 40.0, 20.0)
        mean_perimeter = st.number_input("mean_perimeter", 30.0, 200.0, 90.0)

    with col2:
        mean_area = st.number_input("mean_area", 200.0, 3000.0, 700.0)
        mean_smoothness = st.number_input("mean_smoothness", 0.05, 0.2, 0.1)

    if st.button("Predict"):
        payload = {
            "mean_radius": mean_radius,
            "mean_texture": mean_texture,
            "mean_perimeter": mean_perimeter,
            "mean_area": mean_area,
            "mean_smoothness": mean_smoothness,
        }
        try:
            r = requests.post(f"{API_BASE}/predict", json=payload, timeout=10)
            r.raise_for_status()
            prob = r.json().get("probability", 0.0)
            st.success("Prediction received!")
            st.metric("Cancer Probability", f"{prob:.2%}")
        except requests.exceptions.RequestException as e:
            st.error(f"API request failed: {e}")


# -------------------------------------------------
# Tab 2: Batch Predict
# -------------------------------------------------
with tab_batch:
    st.subheader("Batch Prediction from CSV")

    st.write(
        "Upload a CSV file with the following columns:\n"
        "- mean radius\n"
        "- mean texture\n"
        "- mean perimeter\n"
        "- mean area\n"
        "- mean smoothness\n\n"
        "The app will send all rows to the FastAPI backend and return probabilities."
    )

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
        required_cols = [
            "mean radius",
            "mean texture",
            "mean perimeter",
            "mean area",
            "mean smoothness"
        ]

        if df is not None:
            missing = [c for c in required_cols if c not in df.columns]
            if missing:
                st.error(f"Missing required columns: {missing}")
            else:
                st.write("Preview of uploaded data:")
                st.dataframe(df.head())

                if st.button("Run batch prediction"):
                    records = df[required_cols].to_dict(orient="records")

                    # Map to API field names (underscores instead of spaces)
                    items = [
                        {
                            "mean_radius": r["mean radius"],
                            "mean_texture": r["mean texture"],
                            "mean_perimeter": r["mean perimeter"],
                            "mean_area": r["mean area"],
                            "mean_smoothness": r["mean smoothness"],
                        }
                        for r in records
                    ]

                    try:
                        r = requests.post(
                            f"{API_BASE}/batch_predict", json=items, timeout=60
                        )
                        r.raise_for_status()
                        preds = r.json().get("predictions", [])
                        if not preds:
                            st.warning("No predictions returned.")
                        else:
                            probs = [p["probability"] for p in preds]
                            df_out = df.copy()
                            df_out["probability"] = probs
                            st.success("Batch predictions completed.")
                            st.dataframe(df_out.head())

                            csv_bytes = df_out.to_csv(index=False).encode("utf-8")
                            st.download_button(
                                "Download predictions as CSV",
                                data=csv_bytes,
                                file_name="batch_predictions.csv",
                                mime="text/csv",
                            )
                    except requests.exceptions.RequestException as e:
                        st.error(f"Batch prediction request failed: {e}")
