import streamlit as st, torch, pandas as pd, numpy as np, plotly.graph_objects as go, pybamm

st.set_page_config(page_title="Grok-01 Battery Digital Twin", layout="wide")
st.title("Grok-01: Real-Time Battery Digital Twin")
st.markdown("**Physics-Informed Neural Network** • PyBaMM DFN • Dec 2025")

model = torch.load("models/grok01_pinn.pth", map_location="cpu")
scalers = torch.load("models/scalers.pth")
model.eval()

T = st.slider("Temperature [°C]", -10, 60, 25)
C = st.slider("C-rate", 0.5, 5.0, 1.0, 0.1)

@st.cache_data
def sim(T,C):
    m = pybamm.lithium_ion.DFN()
    p = pybamm.ParameterValues("Chen2020")
    p["Ambient temperature [K]"] = T+273.15
    s = pybamm.Simulation(m, parameter_values=p, C_rate=C)
    return s.solve([0, 3700/C*100])

sol = sim(T,C)
seq = pd.DataFrame({"voltage":sol["Terminal voltage [V]"].entries,
                    "current":sol["Current [A]"].entries,
                    "temperature":sol["Volume-averaged cell temperature [K]"].entries}).values[-100:]
seq = scalers["scaler_X"].transform(seq.reshape(-1,3)).reshape(1,100,3)
pred = scalers["scaler_y"].inverse_transform(model(torch.FloatTensor(seq)).detach().numpy())[0]

col1,col2 = st.columns(2)
col1.metric("Predicted Li⁺ Concentration", f"{pred[0]:,.0f} mol/m³")
col2.metric("Predicted SEI Thickness", f"{pred[1]:.1f} nm")

fig = go.Figure(data=go.Heatmap(z=np.linspace(5000,pred[0],100).reshape(10,10),
                                colorscale="Plasma", colorbar=dict(title="Li⁺")))
fig.update_layout(title="PINN-Predicted Lithium Distribution")
st.plotly_chart(fig, use_container_width=True)
st.success("Live from Colab → Streamlit • Built by SelloP28")
