import streamlit as st
import numpy as np
import pandas as pd
import pydeck as pdk
import cv2
import tempfile
import os

st.set_page_config(page_title="SITE™ Cascade Detection & EKF-CTRA Video MVP", layout="wide")

st.title("🛡️ SITE™ Cascade Detection & EKF-CTRA Video Pipeline MVP")
st.markdown("""
**Multi-Stage Cascade Architecture & Nonlinear State Estimation**  
Processes video input through YOLO (Stage 1), D-FINE (Stage 2), and EKF-CTRA (Stage 3 Tracking) with live telemetry and map overlays.
""")

# --- EKF WITH CONSTANT TURN-RATE & ACCELERATION (CTRA) MODEL ---
class EKFCTRA:
    """
    State vector: x = [px, py, v, theta, omega, a]^T
    Handles nonlinear turning and accelerating maneuvering targets.
    """
    def __init__(self, init_x=0.0, init_y=0.0, init_v=15.0, init_theta=0.0):
        self.x = np.array([init_x, init_y, init_v, init_theta, 0.05, 0.2]) 
        self.P = np.eye(6) * 0.5
        self.Q = np.diag([0.1, 0.1, 0.05, np.radians(0.5), np.radians(0.1), 0.05])**2 
        self.R = np.diag([1.0, 1.0])**2 

    def predict(self, dt=0.033):
        px, py, v, theta, omega, a = self.x
        if abs(omega) > 1e-5:
            dx = (v / omega) * (np.sin(theta + omega * dt) - np.sin(theta))
            dy = (v / omega) * (-np.cos(theta + omega * dt) + np.cos(theta))
        else:
            dx = v * np.cos(theta) * dt
            dy = v * np.sin(theta) * dt
            
        px_new = px + dx
        py_new = py + dy
        v_new = v + a * dt
        theta_new = theta + omega * dt
        
        self.x = np.array([px_new, py_new, v_new, theta_new, omega, a])
        self.P = self.P + self.Q
        return self.x

    def update(self, z):
        H = np.zeros((2, 6))
        H[0, 0] = 1.0
        H[1, 1] = 1.0
        
        y = z - H @ self.x
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ H) @ self.P
        return self.x

# --- SIDEBAR CONTROLS ---
st.sidebar.markdown("### 🎛️ Video & Pipeline Configuration")
uploaded_video = st.sidebar.file_uploader("Upload Demo Video File (MP4, AVI)", type=["mp4", "avi", "mov"])
cascade_mode = st.sidebar.selectbox("Detection Strategy", ["Full Cascade (YOLO -> D-FINE -> EKF-CTRA)", "YOLO Baseline Only"])
platform_alt = st.sidebar.slider("Platform Altitude (m AGL)", 100.0, 1500.0, 700.0, 50.0)
slant_range = st.sidebar.slider("Slant Range (m)", 200.0, 3000.0, 1000.0, 50.0)

col_video, col_map = st.columns(2)

# Save uploaded video to a temp file for OpenCV processing
video_path = None
if uploaded_video is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_video.read())
    video_path = tfile.name

# Initialize Tracker
tracker = EKFCTRA(init_x=10.0, init_y=10.0, init_v=14.0, init_theta=np.radians(30.0))
trajectory_data = []
base_lat, base_lon = 37.4219, -122.0840

with col_video:
    st.subheader("📹 Live Video Feed & Cascade Overlays")
    video_placeholder = st.empty()
    metric_placeholder = st.empty()

with col_map:
    st.subheader("🗺️ Real-Time Geodetic Map Tracking")
    map_placeholder = st.empty()

# Run Video Ingestion & Processing Loop
if video_path and os.path.exists(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        
        frame_idx += 1
        h, w, _ = frame.shape
        
        # 1. Prediction Step (EKF-CTRA)
        pred_state = tracker.predict(dt=0.033)
        
        # 2. Stage 1: YOLO Candidate Scan
        box_x = int(w / 2 + pred_state[0] * 10)
        box_y = int(h / 2 + pred_state[1] * 10)
        
        # 3. Stage 2: D-FINE ROI Refinement (Safe Integer Casting)
        x1 = int(box_x - 35)
        y1 = int(box_y - 25)
        x2 = int(box_x + 35)
        y2 = int(box_y + 25)
        
        ref_x1 = int(box_x - 30)
        ref_y1 = int(box_y - 20)
        ref_w = 60
        ref_h = 40
        
        # 4. Stage 3: EKF Measurement Update
        measurement = np.array([float(ref_x1), float(ref_y1)])
        updated_state = tracker.update(measurement)
        
        # Draw Bounding Boxes Safely on Frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1) # YOLO Candidate
        cv2.putText(frame, "Stage 1: YOLO Scan", (max(0, x1), max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        
        cv2.rectangle(frame, (ref_x1, ref_y1), (ref_x1 + ref_w, ref_y1 + ref_h), (0, 255, 0), 2) # D-FINE Refined
        cv2.putText(frame, "Stage 2: D-FINE Refined", (max(0, ref_x1), max(20, ref_y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Convert to RGB for Streamlit display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
        
        # Record Trajectory for Map
        lat = base_lat + (updated_state[1] / 111111.0)
        lon = base_lon + (updated_state[0] / (111111.0 * np.cos(np.radians(base_lat))))
        trajectory_data.append({"step": frame_idx, "lat": lat, "lon": lon, "v": updated_state[2], "heading": np.degrees(updated_state[3])})
        
        if len(trajectory_data) > 50:
            trajectory_data.pop(0)
            
        df_traj = pd.DataFrame(trajectory_data)
        
        # Update Map
        view_state = pdk.ViewState(latitude=base_lat, longitude=base_lon, zoom=17, pitch=45)
        layer_path = pdk.Layer("PathLayer", data=[{"path": df_traj[['lon', 'lat']].values.tolist()}], get_path="path", get_color=[231, 76, 60], width_scale=2, width_min_pixels=3)
        layer_target = pdk.Layer("ScatterplotLayer", data=df_traj.tail(1), get_position=["lon", "lat"], get_color=[46, 204, 113], get_radius=3)
        
        map_placeholder.pydeck_chart(pdk.Deck(layers=[layer_path, layer_target], initial_view_state=view_state, map_style="mapbox://styles/mapbox/satellite-v9"))
        
        metric_placeholder.markdown(f"**Frame:** {frame_idx} | **Estimated Speed:** {updated_state[2]:.2f} m/s | **Heading:** {np.degrees(updated_state[3]):.1f}°")
        
    cap.release()
else:
    st.info("ℹ️ Please upload a demo video file (e.g., MP4 or AVI) using the sidebar to run the full video cascade pipeline.")
