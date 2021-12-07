"""
In this We will try to make things as customisable as we can  
"""
import streamlit as st
import pandas as pd
import plotly.express as px


@st.cache(persist=True)
def load():
    df = pd.read_csv("avocado.csv")
    df.drop(["Unnamed: 0", "4046", "4225", "4770"], axis=1, inplace=True)
    return df


df = load()

st.title(
    """
    *Avocado Dashboard*
"""
)

st.dataframe(df.sample(10))


def comma_sep(values):
    str_list = " ".join(map(str, values))
    return str_list.split(" ")


st.sidebar.markdown("### Select appropriate params for making graphs")
select_type = st.sidebar.selectbox("Select Type", comma_sep(df["type"].unique()))
select_year = st.sidebar.selectbox("Select Year", comma_sep(df["year"].unique()))
select_region = st.sidebar.selectbox("Select Region", comma_sep(df["region"].unique()))


# def line_graph(year, type, region):
#     graph = st.line_chart(
#         df.loc[
#             ((df["type"] == type) & (df["year"] == year)),
#             ["Total Bags", "Small Bags"],
#         ]
#     )
#     return graph


st.area_chart(
    df.loc[
        ((df["type"] == select_type) & (df["region"] == select_region)),
        ["Total Bags", "Small Bags"],
    ]
)


col1, col2 = st.columns(2)
with col1:
    st.line_chart(
        df.loc[
            ((df["type"] == select_type) & (df["region"] == select_region)),
            ["Total Bags", "Small Bags"],
        ]
    )
with col2:
    st.area_chart(
        df.loc[
            ((df["type"] == select_type) & (df["year"] == int(select_year))),
            ["Total Bags", "Small Bags"],
        ]
    )

st.write("*these are the graphs for total bags and small bags columns*")


px_df = df.loc[
    ((df["type"] == select_type) & (df["year"] == int(select_year))),
    ["XLarge Bags", "Small Bags"],
]

fig = px.bar(px_df)

st.plotly_chart(fig)


col = st.columns(3)

with col[0]:

    st.line_chart(
        df.loc[
            ((df["type"] == select_type) & (df["region"] == select_region)),
            ["Total Bags", "Small Bags"],
        ]
    )

with col[1]:
    st.area_chart(
        df.loc[
            ((df["type"] == select_type) & (df["year"] == int(select_year))),
            ["Total Bags", "Small Bags"],
        ]
    )

with col[2]:
    st.header("An owl")
    st.image("https://static.streamlit.io/examples/owl.jpg")


px_df = df.loc[
    ((df["type"] == select_type) & (df["region"] == select_region)),
    ["Total Bags", "Small Bags"],
]


fig = px.scatter(
    px_df,
    hover_name="name",
    title=f"At {select_region} in {select_year} of type {select_type}",
)

st.plotly_chart(fig)
