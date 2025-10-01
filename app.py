import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# ---------- Functions to generate curves ----------
def generate_curves(L, D):
    x = np.linspace(0, 1, 200)

    # Mocked-up curves depending on L and D
    curve1 = np.tanh(5 * x * D / L)          # Monotonic load-displacement
    curve2 = np.sin(10 * np.pi * x) * D/L    # Cyclic response toy model
    curve3 = np.exp(-x * L / D)              # Degradation curve

    return x, curve1, curve2, curve3

# ---------- Streamlit App ----------
st.title("Monopile Response Explorer")

# Inputs
L = st.number_input("Embedment depth L [m]", min_value=1.0, max_value=100.0, value=20.0, step=1.0)
D = st.number_input("Pile diameter D [m]", min_value=0.5, max_value=20.0, value=5.0, step=0.1)

if st.button("Calculate"):
    x, c1, c2, c3 = generate_curves(L, D)

    # Plot 1
    fig1, ax1 = plt.subplots()
    ax1.plot(x, c1)
    ax1.set_title("Curve 1: Load–Displacement (mock)")
    ax1.set_xlabel("Displacement (normalized)")
    ax1.set_ylabel("Load (normalized)")
    st.pyplot(fig1)

    # Plot 2
    fig2, ax2 = plt.subplots()
    ax2.plot(x, c2)
    ax2.set_title("Curve 2: Cyclic Response (mock)")
    ax2.set_xlabel("Cycle fraction")
    ax2.set_ylabel("Response amplitude")
    st.pyplot(fig2)

    # Plot 3
    fig3, ax3 = plt.subplots()
    ax3.plot(x, c3)
    ax3.set_title("Curve 3: Stiffness Degradation (mock)")
    ax3.set_xlabel("Cycles (normalized)")
    ax3.set_ylabel("Relative stiffness")
    st.pyplot(fig3)