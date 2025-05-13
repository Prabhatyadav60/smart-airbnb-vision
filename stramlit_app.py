# app.py
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from ultralytics import YOLO
import av
import numpy as np

# Load your custom YOLOv8 model once
@st.cache_resource
def load_model():
    return YOLO("best.pt")
model = load_model()

st.title("📹 Real-time YOLOv8 Detection in Streamlit")

# Sidebar controls
conf_thresh = st.sidebar.slider(
    "Confidence Threshold", 0.0, 1.0, 0.25, 0.01, key="conf"
)
st.sidebar.markdown("---")
st.sidebar.write("⚙️ Adjust threshold to filter weak detections.")

class Detector(VideoTransformerBase):
    def __init__(self):
        self.model = model
        self.conf = conf_thresh

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        # Convert to numpy BGR image
        img = frame.to_ndarray(format="bgr24")

        # Run inference
        results = self.model(img, conf=self.conf)

        # Annotate frame
        annotated = results[0].plot()

        # Convert back to VideoFrame
        return av.VideoFrame.from_ndarray(annotated, format="bgr24")

webrtc_streamer(
    key="yolo-webcam",
    video_transformer_factory=Detector,
    media_stream_constraints={"video": True, "audio": False},
    async_transform=True,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
)
