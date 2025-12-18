import streamlit as st
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances_argmin

st.set_page_config(page_title="Hierarchical Clustering App", layout="centered")

st.title("📊 Hierarchical Clustering App")
st.caption("Agglomerative Clustering (Ward linkage)")

# ---------- Load saved objects ----------
# Assumes you saved these during training
with open("hc_objects.pkl", "rb") as f:
    scaler, cluster_centroids, data = pickle.load(f)

st.divider()

# ---------- User Input ----------
st.subheader("Enter Customer Details")

gender = st.selectbox("Gender", ["Male", "Female"])
region = st.selectbox("Region", ["Urban", "Rural"])
income = st.number_input("Income", min_value=0.0, max_value=100.0, step=1.0)
spending = st.number_input("Spending", min_value=0.0, max_value=50.0, step=1.0)

# Manual encoding (must match training)
gender_encoded = 1 if gender == "Male" else 0
region_encoded = 1 if region == "Urban" else 0

if st.button("Find Cluster"):
    user_data = np.array([[gender_encoded, region_encoded, income, spending]])
    user_scaled = scaler.transform(user_data)

    cluster_label = pairwise_distances_argmin(
        user_scaled, cluster_centroids
    )[0]

    st.success(f"User belongs to **Cluster {cluster_label}**")

    # ---------- Visualization ----------
    fig, ax = plt.subplots()
    ax.scatter(
        data["Income"],
        data["Spending"],
        c=data["Cluster"]
    )
    ax.scatter(
        income,
        spending,
        color="red",
        s=120,
        marker="X",
        label="User"
    )
    ax.set_xlabel("Income")
    ax.set_ylabel("Spending")
    ax.set_title("Hierarchical Clustering Result")
    ax.legend()

    st.pyplot(fig)

st.divider()
st.caption("Hierarchical Clustering • Streamlit")
