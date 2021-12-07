from importlib import import_module
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objs as go
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator


st.set_option('deprecation.showPyplotGlobalUse', False)

st.markdown(
    """
    <style>
    .reportview-container {
        background:#000000
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
    df = pd.read_csv("avocado.csv", parse_dates=['Date'])
    df.drop(["Unnamed: 0", "4046", "4225", "4770"], axis=1, inplace=True)
    return df


df = load()
st.write(df.sample(20))

chart = px.bar(df, x='Date', y='AveragePrice', color='type', facet_col='type')

st.write(chart)

df_types = df.groupby(['type', 'year'])['Total Volume'].sum()


'''
format for customising graphs
'''
# fig = go.Figure()
# fig.add_traces(go.Bar(
#     df[['year', 'Total Bags']])),  # , y=df['Total Bags'], mode='lines',
# # name='lines'))
# fig.update_layout(
#     title={
#         'text': "Plot Title",
#         # 'y': 1,
#         # 'x': 0.5,
#         'xanchor': 'center',
#         'yanchor': 'top'},
#     xaxis=dict(
#         barmode='stack',
#         tickformat=",d",
#         # tickvals=[1, 3, 5, 7],
#         ticktext=[2015, 2016, 2017, 2018]
#     ),
#     font_family="Courier New",
#     font_color="blue",
#     title_font_family="Times New Roman",
#     title_font_color="white",
#     legend_title_font_color="green", xaxis_title="Year Of Purchases")


# line chart

fig = px.line(df, x='Date', y='AveragePrice', facet_col='type', color='type', color_discrete_map={
    'conventional': "#FD833E",
    # 'organic': "  #ffff80"
})

fig.data[1].line.color = '#8080ff'
fig.update_layout(
    title_text='Total Bags PerYear of Different types of Avocado',
    title_x=0.5, font_color="black",
    title_font_color="white",
    legend_title_font_color="green")

# fig.data[1].grid = True
fig.update_xaxes(showline=True, linewidth=2,
                 linecolor='black', gridcolor='black')
fig.update_yaxes(showline=True, linewidth=2,
                 linecolor='black', gridcolor='black')


# fig.layout.yaxis.gridcolor = 'black'
# fig.layout.xaxis.gridcolor = 'black'
fig.layout.plot_bgcolor = '#fff'
fig.layout.paper_bgcolor = '#fff'
st.plotly_chart(fig)

# fig = go.Figure()

# fig.add_traces(go.Histogram())

hist_data = df.groupby(['type'])['AveragePrice'].sum()
print(hist_data)
fig, ax = plt.subplots(1, 2)
ax[0] = sns.distplot(df.loc[df['type'] == 'conventional',
                            'AveragePrice'], color='#ff80ff', hist=False, ax=ax[0])
ax[1] = sns.distplot(df.loc[df['type'] == 'organic',
                            'AveragePrice'], color='#8080ff', hist=False, ax=ax[1], label=['organic'], axlabel={'x': ['Average Prices'], 'y': None})
ax[0].set_title = 'conventional'
ax[0].set_ylabel = None
ax[1].set_ylabel = 'Average Prices'
ax[0].grid(True)
ax[1].grid(True)

ax[1].suptitle = 'organic'
plt.title('plot of Average prices for types of avocados', loc='center')

st.pyplot(fig)


# seaborn approach

fig, ax = plt.subplots(1, 1)
sns.kdeplot(df.loc[df['type'] == 'conventional', 'year'],
            df.loc[df['type'] == 'conventional', 'AveragePrice'], shade=True)
# ax = plt.figure().gca()
ax.title.set_text('Average Price with year [not in int xticks]')
ax.ticklabel_format(useOffset=False)
# plt.xticks(df['year'].astype(int))

st.write(fig)
# graph number 3 ---- 4 subplotsbetween types of avocados and total volume,average prices

# I need to group this data month wise for having a good rep. of the data
# so lets convert
print(df.info())

df.set_index('Date')
df['Date'] = pd.DatetimeIndex(df)
st.write(df.head())
monthly_av = df.groupby(['AveragePrice', 'Date'],
                        pd.Grouper(key='Date', freq='M'))


fig, ax = plt.subplots(2, 2, figsize=(20, 20))

ax[0, 0] = plt.plot('year', 'AveragePrice', data=df.loc[df['type']
                                                        == 'conventional'], label=['dates'], color='#1a75ff')
st.pyplot(fig)
