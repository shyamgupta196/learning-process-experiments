'''
This is an exact replica of the kaggle notebooks
on avocados
'''
from scipy.interpolate import make_interp_spline
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
import numpy as np
import matplotlib.dates as mdates


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

# removing spines is a tedious task so i make a function to remove these


def remove_spines(ax):
    ax[0, 0].spines['top'].set_visible(False)
    ax[0, 0].spines['right'].set_visible(False)
    ax[0, 0].spines['left'].set_visible(False)
    ax[1, 0].spines['top'].set_visible(False)
    ax[1, 0].spines['right'].set_visible(False)
    ax[1, 0].spines['left'].set_visible(False)
    ax[0, 1].spines['top'].set_visible(False)
    ax[0, 1].spines['right'].set_visible(False)
    ax[0, 1].spines['left'].set_visible(False)
    ax[1, 1].spines['top'].set_visible(False)
    ax[1, 1].spines['right'].set_visible(False)
    ax[1, 1].spines['left'].set_visible(False)


# density plot seaborn -> NB graph 1
fig, ax = plt.subplots(1, 2, sharey=True)
sns.kdeplot(df.loc[df['type'] == 'conventional', 'AveragePrice'],
            color='#00cc7a', ax=ax[0], shade=True, alpha=1)

ax[0].spines[['top', 'right', 'left']].set_visible(False)
ax[1].spines[['top', 'right', 'left']].set_visible(False)


sns.kdeplot(df.loc[df['type'] == 'organic', 'AveragePrice'],
            color='#ff8533', ax=ax[1], shade=True, alpha=1)
ax[0].legend(['conventional'], loc='upper left')
ax[1].legend(['organic'], loc='upper right')
ax[0].grid()
fig.text(0.5, 0.04, 'Average Price Avocados', ha='center', va='center')
ax[1].grid()


# this is another way of adding different things
# to both subplots
for i in ax:
    # i.grid()
    i.set_xlabel('')

st.write(fig)

# this is Line graph from kaggle NB
# the Only reason this graph is a bit different from the NB graph is that
# the df is grouped into months in the notebook
# hence they have a neat graph *thats it*

chart = px.line(df, x='Date', y='AveragePrice', color='type', color_discrete_map={
    'organic': '#ff6666', 'conventional': '#47d1d1'}, facet_col='type')
chart.update_layout(
    title_text='Total Bags PerYear of Different types of Avocado',
    title_x=0.5, font_color="black",
    title_font_color="white",
    legend_title_font_color="green",
    paper_bgcolor='white',
    plot_bgcolor='white',
    grid_pattern='coupled',
    legend_valign='bottom',
    legend=dict(
        yanchor="bottom",
        x=0.5,
        xanchor="center",
        y=-0.3
    ),
    xaxis_title="Date"
)
chart.update_xaxes(showgrid=True, gridcolor='grey', gridwidth=1.5)
chart.update_yaxes(showgrid=True, gridcolor='grey', gridwidth=1.5)

st.write(chart)

# 4subplots complex graphs
# firstly make data into months
# group them


conventional = df.loc[df['type'] == 'conventional', :]
organic = df.loc[df['type'] == 'organic', :]

conv_counts = conventional.groupby([pd.Grouper(key='Date', axis=0,
                                               freq='M'), 'type'])['AveragePrice'].count()
org_counts = organic.groupby([pd.Grouper(key='Date', axis=0,
                                         freq='M'), 'type'])['AveragePrice'].count()

conv_grp = conventional.groupby([pd.Grouper(key='Date', axis=0,
                                            freq='M'), 'type'])['AveragePrice', 'Total Volume'].mean()
org_grp = organic.groupby([pd.Grouper(key='Date', axis=0,
                                      freq='M'), 'type'])['AveragePrice', 'Total Volume'].mean()
org_grp = org_grp.reset_index()

# plot 3

fig, ax = plt.subplots(2, 2, facecolor='#a6a6a6')
org_grp = pd.DataFrame({'Date': org_grp['Date'].dt.date,
                        'type': org_grp['type'].values,
                        'AveragePrice': org_grp['AveragePrice'],
                        'Total Volume': org_grp['Total Volume']
                        })

conv_grp = conv_grp.reset_index()

conv_grp = pd.DataFrame({'Date': conv_grp['Date'].dt.date,
                         'type': conv_grp['type'].values,
                         'AveragePrice': conv_grp['AveragePrice'],
                         'Total Volume': conv_grp['Total Volume']})


conv_grp[['Date', 'Total Volume']].plot(ax=ax[1, 0], kind='bar')
conv_grp[['Date', 'Total Volume']].plot(ax=ax[1, 0], color='red', alpha=0.7)


ax[0, 0].plot(conv_grp['Date'], conv_grp['AveragePrice'])
ax[0, 0].set_xticklabels([], rotation='45')
ax[0, 0].grid(axis='y', linewidth=2, color='white')
ax[0, 0].set_facecolor('#a6a6a6')

remove_spines(ax)

ax[0, 1].plot(org_grp['Date'], org_grp['AveragePrice'], color='green')
ax[0, 1].set_xticklabels([], rotation='45')
ax[0, 1].grid(axis='y', linewidth=2, color='white')
ax[0, 1].set_facecolor('#a6a6a6')


ax[1, 0].set_yticklabels([0, 50000, 100000, 150000, 200000, 250000])
ax[1, 0].grid(axis='y', linewidth=2, color='white')
ax[1, 0].set_facecolor('#a6a6a6')
ax[1, 0].get_legend().remove()


org_grp[['Date', 'Total Volume']].plot(ax=ax[1, 1], kind='bar', color='green')
org_grp[['Date', 'Total Volume']].plot(ax=ax[1, 1], color='red', alpha=0.7)

ax[1, 1].set_yticklabels([0, 50000, 100000, 150000, 200000, 250000])
ax[1, 1].get_legend().remove()
ax[1, 1].grid(axis='y', linewidth=2, color='white')
ax[1, 1].set_facecolor('#a6a6a6')
st.write(fig)


fig, axes = plt.subplots(2, 2, sharey=True, facecolor='#ffcc66')


sns.kdeplot(df.loc[df['year'] == 2015, 'AveragePrice'],
            color='#8080ff', ax=axes[0, 0], shade=True)


axes[0, 0].set_xlabel('')
axes[0, 1].set_xlabel('')
axes[1, 1].set_xlabel('')
axes[1, 0].set_xlabel('')

axes[0, 0].set_facecolor('#ffcc66')
axes[0, 1].set_facecolor('#ffcc66')
axes[1, 1].set_facecolor('#ffcc66')
axes[1, 0].set_facecolor('#ffcc66')

axes[0, 0].grid(axis='y', linewidth=2, color='white')
axes[0, 1].grid(axis='y', linewidth=2, color='white')
axes[1, 0].grid(axis='y', linewidth=2, color='white')
axes[1, 1].grid(axis='y', linewidth=2, color='white')

axes[0, 0].set_title('2015')
axes[0, 1].set_title('2016')
axes[1, 0].set_title('2017')
axes[0, 1].set_title('2018')

plt.title('Distribution of Prices By Year')

sns.kdeplot(df.loc[df['year'] == 2016, 'AveragePrice'],
            color='#88ff4d', ax=axes[0, 1], shade=True)
sns.kdeplot(df.loc[df['year'] == 2017, 'AveragePrice'],
            color='#ff944d', ax=axes[1, 0], shade=True)
sns.kdeplot(df.loc[df['year'] == 2018, 'AveragePrice'],
            color='#ff4d4d', ax=axes[1, 1], shade=True)
remove_spines(axes)
fig.suptitle('Distribution of prices by year')
st.write(fig)
