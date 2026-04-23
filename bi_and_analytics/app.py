import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.datasets import load_iris

# 1. Page Configuration (Visualization Layer)
st.set_page_config(page_title="Iris Insights Dashboard", layout="wide")

# 2. Ingestion & Transformation Layer (with Caching)
@st.cache_data
def load_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df["species"] = iris.target
    df["species"] = df["species"].map(dict(enumerate(iris.target_names)))
    return df, iris.feature_names

df, features = load_data()

# 3. Sidebar Controls (Interactivity Layer)
st.sidebar.header("Filter & Configure")
species_selected = st.sidebar.multiselect(
    "Select Species:", options=df["species"].unique(), default=df["species"].unique()
)

x_axis = st.sidebar.selectbox("X-axis Variable:", options=features, index=0)
y_axis = st.sidebar.selectbox("Y-axis Variable:", options=features, index=1)

# Filter Logic
filtered_df = df[df["species"].isin(species_selected)]

# 4. Main Dashboard (Visualization Layer)
st.title("🌸 Iris Dataset Explorer")
st.markdown("Explore relationships and distributions within the classic Iris botanical dataset.")

# KPI Row
col1, col2, col3 = st.columns(3)
col1.metric("Total Observations", len(filtered_df))
col2.metric("Species Selected", len(species_selected))
col3.metric("Avg Sepal Length", f"{filtered_df[features[0]].mean():.2f} cm")

# Interactive Plotly Chart
fig = px.scatter(
    filtered_df, x=x_axis, y=y_axis, color="species",
    title=f"Relationship: {x_axis} vs {y_axis}",
    template="plotly_white", hover_data=features
)
st.plotly_chart(fig, width="stretch")

# Distribution Deep-Dive
st.subheader("Feature Distribution")
feature_to_plot = st.selectbox("Select Feature for Histogram:", options=features)
fig_hist = px.histogram(filtered_df, x=feature_to_plot, color="species", barmode="overlay")
st.plotly_chart(fig_hist, width="stretch")