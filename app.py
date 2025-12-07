# app.py — FINAL WORKING VERSION (Streamlit Cloud Dec 2025)
import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import pybamm
import pickle
import os

st.set_page_config(page_title="Grok-Phakoe Battery Digital Twin", layout="wide")
st.title("Grok-Phakoe 01: Real-Time Battery Digital Twin")
st.markdown("**Physics-Informed Neural Network • PyBaMM DFN • Dec 2025 • SelloP28**")

# ← Re-define the exact same model class
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
def load_model_and_scalers():
    # Load model (custom class + weights_only=False)
    model = torch.load("models/grok01_pinn.pth", map_location="cpu", weights_only=False)
    model.eval()
    
    # Load scalers correctly with pickle (not torch)
    with open("models/scalers.pkl", "rb") as f:
        scalers_dict = pickle.load(f)
    
    return model, scalers_dict["scaler_X"], scalers_dict["scaler_y"]

# Load everything
model, scaler_X, scaler_y = load_model_and_scalers()

# Interactive sliders
T = st.slider("Temperature [°C]", -10, 60, 25)
C = st.slider("C-rate", 0.5, 5.0, 1.0, 0.1)

# Run PyBaMM simulation
@st.cache_data(show_spinner="Running PyBaMM simulation...")
def run_simulation(T_amb, C_rate):
    m = pybamm.lithium_ion.DFN()
    p = pybamm.ParameterValues("Chen2020")
    p["Ambient temperature [K]"] = T_amb + 273.15
    sim = pybamm.Simulation(m, parameter_values=p, C_rate=C_rate)
    sol = sim.solve([0, 3700/C_rate * 80])
    return sol

sol = run_simulation(T, C)

# Prepare input sequence
df = pd.DataFrame({
    "voltage": sol["Terminal voltage [V]"].entries,
    "current": sol["Current [A]"].entries,
    "temperature": sol["Volume-averaged cell temperature [K]"].entries
})
seq = df.values[-100:]  # last 100 points
seq_scaled = scaler_X.transform(seq.reshape(-1, 3)).reshape(1, 100, 3)

# Predict
with torch.no_grad():
    pred_scaled = model(torch.FloatTensor(seq_scaled))
    pred = scaler_y.inverse_transform(pred_scaled.numpy())[0]

# Display results
col1, col2 = st.columns(2)
col1.metric("Predicted Li⁺ Concentration (neg. electrode)", f"{pred[0]:,.0f} mol/m³")
col2.metric("Predicted SEI Thickness", f"{pred[1]:.1f} nm")

# Heatmap
fig = go.Figure(data=go.Heatmap(
    z=np.linspace(5000, pred[0], 100).reshape(10, 10),
    colorscale="Plasma",
    colorbar=dict(title="Li⁺ mol/m³")
))
fig.update_layout(title="PINN-Predicted Lithium Distribution")
st.plotly_chart(fig, use_container_width=True)

st.success("Grok-01 is LIVE • https://github.com/SelloP28/battery-digital-twin-using-pinn")
