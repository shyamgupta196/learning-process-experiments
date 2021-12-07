import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# from Inputs_Parallel import get_possible_scenarios

# Side Bar #######################################################
project_title = st.sidebar.text_input(label="Title of Project", value="Example Project")

username = st.sidebar.selectbox(label="Username", options=("a_name", "b_name"))

buildable_land_folder = st.sidebar.text_input(
    label="Buildable Land Folder", value=r"\\filepath\example"
)

config_file_location = st.sidebar.text_input(
    label="Config File", value=r"\\filepath\example"
)

gcr_config = st.sidebar.slider(
    label="Ground Coverage Ratio Range Selection",
    min_value=10,
    max_value=60,
    step=1,
    value=(28, 45),
)

sr_config = st.sidebar.slider(
    label="Sizing Ratio Range Selection",
    min_value=1.0,
    max_value=2.0,
    step=0.1,
    value=(1.0, 1.5),
)

run_button = st.sidebar.button(label="Run Optimization")

progress_bar = st.sidebar.progress(0)

# App ###########################################################
st.title(project_title)

# Graphing Function #####
z_data = pd.read_csv(
    "https://raw.githubusercontent.com/plotly/datasets/master/api_docs/mt_bruno_elevation.csv"
)
z = z_data.values
sh_0, sh_1 = z.shape
x, y = np.linspace(0, 1, sh_0), np.linspace(0, 1, sh_1)
fig = go.Figure(data=[go.Surface(z=z, x=x, y=y)])
fig.update_layout(
    title="IRR",
    autosize=False,
    width=800,
    height=800,
    margin=dict(l=40, r=40, b=40, t=40),
)
st.plotly_chart(fig)
