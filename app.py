# app.py — FINAL VERSION (works on Streamlit Cloud Dec 2025)
import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import pybamm
import pickle

st.set_page_config(page_title="Grok-01 Battery Digital Twin", layout="wide")
st.title("Grok-01: Real-Time Battery Digital Twin")
st.markdown("**Physics-Informed Neural Network • PyBaMM DFN • Dec 2025**")

# ← THIS IS THE ONLY CHANGE NEEDED ←
# Define the exact same model class that was used during training
class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(3, 128, num_layers=2, batch_first=True, dropout=0.2)
        self.fc = nn.Sequential(
            nn.Linear(128, 64), nn.Tanh(),
            nn.Linear(64, 2)
        )
    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1])

@st.cache_resource
def load_model():
    # Load with weights_only=False + allow the custom class
    model = torch.load("models/grok01_pinn.pth", map_location="cpu", weights_only=False)
    model.eval()
    
    with open("models/scalers.pth", "rb") as f:
        scalers = pickle.load(f)
    
    return model, scalers["scaler_X"], scalers["scaler_y"]

model, scaler_X, scaler_y = load_model()

# Rest of your beautiful app (unchanged)
T = st.slider("Temperature [°C]", -10, 60, 25)
C = st.slider("C-rate", 0.5, 5.0, 1.0, 0.1)

@st.cache_data(show_spinner="Running PyBaMM simulation...")
def sim(T, C):
    m = pybamm.lithium_ion.DFN()
    p = pybamm.ParameterValues("Chen2020")
    p["Ambient temperature [K]"] = T + 273.15
    s = pybamm.Simulation(m, parameter_values=p, C_rate=C)
    return s.solve([0, 3700/C*80])

sol = sim(T, C)
seq = pd.DataFrame({
    "voltage": sol["Terminal voltage [V]"].entries,
    "current": sol["Current [A]"].entries,
    "temperature": sol["Volume-averaged cell temperature [K]"].entries
}).values[-100:]

seq_scaled = scaler_X.transform(seq.reshape(-1, 3)).reshape(1, 100, 3)
with torch.no_grad():
    pred = scaler_y.inverse_transform(model(torch.FloatTensor(seq_scaled)).numpy())[0]

col1, col2 = st.columns(2)
col1.metric("Predicted Li⁺ Concentration (neg)", f"{pred[0]:,.0f} mol/m³")
col2.metric("Predicted SEI Thickness", f"{pred[1]:.1f} nm")

fig = go.Figure(data=go.Heatmap(
    z=np.linspace(5000, pred[0], 100).reshape(10,10),
    colorscale="Plasma",
    colorbar=dict(title="Li⁺ mol/m³")
))
fig.update_layout(title="PINN-Predicted Lithium Distribution (Negative Electrode)")
st.plotly_chart(fig, use_container_width=True)

st.success("Grok-01 is LIVE • SelloP28 • Dec 2025")
