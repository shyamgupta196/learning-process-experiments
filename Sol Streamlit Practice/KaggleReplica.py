from importlib import import_module
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go

st.markdown(
    """
    <style>
    .reportview-container {
        background:#ffffff
    }
   .sidebar .sidebar-content {
        background: #4d79ff
    }
    .font{
        color: "black",
        }
    </style>
    """,
    unsafe_allow_html=True)

st.sidebar.selectbox("Select Graph", ['Scatter', 'Bar', 'Line'])


@st.cache(persist=True)
def load():
    df = pd.read_csv("avocado.csv")
    df.drop(["Unnamed: 0", "4046", "4225", "4770"], axis=1, inplace=True)
    return df


df = load()
st.write(df.sample(20))

chart = px.area(df, 'Date', 'AveragePrice', color='type')

st.write(chart)

df_types = df.groupby(['type', 'year'])['Total Volume'].sum()

# make something with conventional and organic in same year multirow index
# size and position of pie chart to be on the header
# mking horizontal divisions for the nav bar ///// OR //// making a nav bar if possible !
# using plotly.go for more beautiful customisations , its tough though but i have to learn it anyway!!


st.write(df_types)
chart = px.pie(df_types, 'Total Volume')
st.write(chart)

# also caching graphs with a function
# the number of colors will be equal to
# number of unique values in the pie........

fig = go.Figure(data=[go.Pie(labels=df['type'], values=df['Total Volume'])])
fig.update_traces(hoverinfo='label+percent', textinfo='value', textfont_size=20,
                  marker=dict(line=dict(color='#4d79ff', width=5)))
st.plotly_chart(fig)
