
import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import joblib

from pathlib import Path
from PIL import Image

st.set_page_config(
    page_title="SignSense AI",
    page_icon="🤟",
    layout="wide"
)

st.title("🤟 SignSense AI")
st.subheader("Machine Learning Hand Gesture Recognition")
st.write("Capture a hand gesture to classify it using Random Forest.")

MODEL_PATH = Path(__file__).parent / "gesture_model.pkl"


@st.cache_resource
def load_model(path, modified_time):
    saved = joblib.load(path)
    return saved["model"], saved["features"]


if not MODEL_PATH.exists():
    st.error("Model not found. Run train_model.py first.")
    st.stop()

try:
    model, feature_names = load_model(
        str(MODEL_PATH), MODEL_PATH.stat().st_mtime
    )
except Exception as error:
    st.error(f"Unable to load model: {error}")
    st.stop()

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils


def extract_features(hand_landmarks):
    points = np.array(
        [[p.x, p.y, p.z] for p in hand_landmarks.landmark],
        dtype=np.float32
    )

    # Must match collect_data.py exactly
    points -= points[0].copy()

    scale = np.max(np.abs(points[:, :2]))
    if scale < 1e-6:
        return None

    points /= scale
    return points.flatten().tolist()


if "history" not in st.session_state:
    st.session_state.history = []

photo = st.camera_input("Take a picture of your hand")

if photo is not None:
    frame = np.array(Image.open(photo).convert("RGB"))

    # Collection used cv2.flip(frame, 1)
    frame = cv2.flip(frame, 1)

    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.6
    ) as hands:
        result = hands.process(frame)

    if not result.multi_hand_landmarks:
        st.warning("No hand detected. Improve lighting and try again.")
    else:
        hand = result.multi_hand_landmarks[0]
        features = extract_features(hand)

        if features is None:
            st.warning("Could not extract usable hand landmarks.")
        else:
            X = pd.DataFrame([features], columns=feature_names)
            prediction = model.predict(X)[0]
            probabilities = model.predict_proba(X)[0]
            confidence = float(np.max(probabilities)) * 100

            annotated = cv2.cvtColor(frame.copy(), cv2.COLOR_RGB2BGR)
            mp_draw.draw_landmarks(
                annotated, hand, mp_hands.HAND_CONNECTIONS
            )
            annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

            left, right = st.columns(2)

            with left:
                st.image(
                    annotated,
                    caption="Detected hand landmarks",
                    use_container_width=True
                )

            with right:
                st.success(f"Predicted gesture: {prediction}")
                st.metric("Model confidence", f"{confidence:.1f}%")

                probability_df = pd.DataFrame({
                    "Gesture": model.classes_,
                    "Probability (%)": probabilities * 100
                }).sort_values("Probability (%)", ascending=False)

                st.bar_chart(probability_df.set_index("Gesture"))

            st.session_state.history.append({
                "Gesture": prediction,
                "Confidence (%)": round(confidence, 1)
            })

st.divider()
st.subheader("Prediction History")

if st.session_state.history:
    st.dataframe(
        pd.DataFrame(st.session_state.history),
        use_container_width=True,
        hide_index=True
    )

    if st.button("Clear History"):
        st.session_state.history = []
        st.rerun()
else:
    st.caption("Your predictions will appear here.")
