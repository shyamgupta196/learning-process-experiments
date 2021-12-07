import streamlit as st
import pandas as pd
import numpy as np

## CLEANING THE DATA FOR EDA AND DASHBOARD

DATA_URL = "avocado.csv"


@st.cache(
    persist=True
)  # ( If you have a different use case where the data does not change so very often, you can simply use this)
def load_data():
    df = pd.read_csv(DATA_URL)
    df.drop(["Unnamed: 0", "4046", "4225", "4770", "XLarge Bags"], axis=1, inplace=True)
    return df


df = load_data()


for i in df.columns:
    print(df[f"{i}"].unique())

st.title(
    """
 **Avacado Dashboard**
 """
)

# st.dataframe(df.style.highlight_max(axis=0))


if st.button("Welcome"):
    st.write("Hello SOL :wave:")
else:
    st.write("Click Here")

st.dataframe(df.head())
select_type = st.sidebar.selectbox(
    "Types",
    [
        "conventional",
        "organic",
    ],
    key="1",
)


select_year = st.sidebar.selectbox(
    "year",
    [2015, 2016, 2017, 2018],
    key="1",
)

if not st.sidebar.checkbox("Hide", True, key="1"):

    if select_type == "conventional":
        st.line_chart(
            df.loc[
                ((df["type"] == "conventional") & (df["year"] == select_year)),
                ["Total Bags", "Small Bags"],
            ]
        )
    if select_type == "organic":
        st.line_chart(
            df.loc[
                ((df["type"] == "organic") & (df["year"] == select_year)),
                ["Total Bags", "Small Bags"],
            ]
        )
# apply filters for region toooo but thats a tedious process to do manually soo we just


# type
# ['conventional' 'organic']

# region
# ['Albany' 'Atlanta' 'BaltimoreWashington' 'Boise' 'Boston'
#  'BuffaloRochester' 'California' 'Charlotte' 'Chicago' 'CincinnatiDayton'
#  'Columbus' 'DallasFtWorth' 'Denver' 'Detroit' 'GrandRapids' 'GreatLakes'
#  'HarrisburgScranton' 'HartfordSpringfield' 'Houston' 'Indianapolis'
#  'Jacksonville' 'LasVegas' 'LosAngeles' 'Louisville' 'MiamiFtLauderdale'
#  'Midsouth' 'Nashville' 'NewOrleansMobile' 'NewYork' 'Northeast'
#  'NorthernNewEngland' 'Orlando' 'Philadelphia' 'PhoenixTucson'
#  'Pittsburgh' 'Plains' 'Portland' 'RaleighGreensboro' 'RichmondNorfolk'
#  'Roanoke' 'Sacramento' 'SanDiego' 'SanFrancisco' 'Seattle'
#  'SouthCarolina' 'SouthCentral' 'Southeast' 'Spokane' 'StLouis' 'Syracuse'
#  'Tampa' 'TotalUS' 'West' 'WestTexNewMexico']

# year
# [2015 2016 2017 2018]
