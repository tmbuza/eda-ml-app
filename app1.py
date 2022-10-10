import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from numpy.core.numeric import True_
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score


#--------------------------------
# PAGE CONFIG
#--------------------------------
st.set_page_config(
     page_title='ML WebApps',
     page_icon=':bar_chart:',
     layout='centered',
     initial_sidebar_state='expanded')

#--------------------------------
# ASSETS
#--------------------------------
testdata = pd.read_csv("data/breast_cancer.csv")
diamonds = pd.read_csv("data/preprocessed_diamonds.csv")
clean = Image.open("img/ml.png")
cover = Image.open("img/diamonds.png",)

#--------------------------------
# PAGE HEADER
#--------------------------------
def main():
    with st.container():
      st.markdown("<h1 style='text-align: center; color: grey;'>Machine Learning Web Applications using Python and Streamlit Library</h1>", unsafe_allow_html=True)
      st.write("##")
      st.write("##")
      st.success(
        """ 
        # :shower: Data Preprocessing App
        """)
      st.image(cover)
      st.markdown(
        """
        ### App Functionalities
        - This App helps you to create a tidy dataframe for downstream machine learning problems.
          - Reads the `data` uploaded by user and automatically starts exploration.
          - Displays the `head` and `tail` of the uploaded data (input dataframe).
          - Shows the dataframe `dimension`, `variable names` and `missing values`.
          - Perform `descriptive` statistical analysis of the numerical variables.
          - Plots a `correlation` heatmap.
        """)
        
        
    # with panel3:
      st.write("""
        ### Testing the App""")
      st.markdown(
        """
        - You may start by testing the app using a demo data by clicking on the `Test App` button, otherwise:
        - Upload a tidy input file using the user\'s widget on the left sidebar. 
        """)

  
if __name__ == '__main__':
  main()
  
st.write("---")


with st.sidebar.header('User input widget'):
    uploaded_file = st.sidebar.file_uploader("Please choose a tidy CSV file", type=["csv"])

st.header("Starting Data Exploration")

if st.button('Click to test the App using a demo dataset!'):
  # Using a demo data
  @st.cache
  def load_data():
      a = pd.read_csv('data/preprocessed_diamonds.csv')
      return a

  df = load_data()
  st.write(""" > This demo uses a preprocessed `diamond dataset` for demonstration only.""")
  st.header("""Demo Data Exploration""")
  st.subheader("""Input DataFrame""")
  st.write("Head", df.head())
  st.write("Tail", df.tail())

  st.subheader("""Dataframe dimension""")
  st.markdown("> Note that 0 = rows, and 1 = columns")
  st.dataframe(df.shape)

  st.subheader("""Variable names""")
  st.dataframe(pd.DataFrame({'Variable': df.columns}))

  st.subheader("""Missing values""")
  missing_count = df.isnull().sum()
  value_count = df.isnull().count() #denominator
  missing_percentage = round(missing_count/value_count*100, 2)
  missing_df = pd.DataFrame({'Count': missing_count, 'Missing (%)': missing_percentage})
  st.dataframe(missing_df)

  st.header("""Basic Statistics""")
  st.subheader("""Descriptive statistics""")
  st.dataframe(df.describe())
  
  st.write("##")
  st.subheader("""Scatter diagram""")  
  fig, ax = plt.subplots(1,1)
  sns.scatterplot(x=df.iloc[:, 0], y=df.iloc[:, 6], hue = df.iloc[:, 3]) # Change positional index to match the location of variables.
  # ax.set_xlabel('X-axis values')
  # ax.set_ylabel('Y-axis values')
  st.pyplot(fig)
  
  st.write("##")
  st.write("##")
  st.subheader("""Correlation heatmap""")
  fig, ax = plt.subplots()
  sns.heatmap(df.corr(), ax = ax)
  st.write(fig, use_container_width=False)
  
  
if uploaded_file is not None:
  def load_data():
      a = pd.read_csv(uploaded_file)
      return a
  df = load_data()
  st.subheader("""Input DataFrame""")
  st.write("Head", df.head())
  st.write("Tail", df.tail())

  st.subheader("""Dataframe dimension""")
  st.markdown("> Note that 0 = rows, and 1 = columns")
  st.dataframe(df.shape)

  st.subheader("""Variable names""")
  st.dataframe(pd.DataFrame({'Variable': df.columns}))

  st.subheader("""Missing values""")
  missing_count = df.isnull().sum()
  value_count = df.isnull().count() #denominator
  missing_percentage = round(missing_count/value_count*100, 2)
  missing_df = pd.DataFrame({'Count': missing_count, 'Missing (%)': missing_percentage})
  st.dataframe(missing_df)

  st.header("""Basic Statistics""")
  st.subheader("""Descriptive statistics""")
  st.dataframe(df.describe())
  
  # st.write("##")
  # st.subheader("""Scatter diagram""")  
  # fig, ax = plt.subplots(1,1)
  # sns.scatterplot(x=df.iloc[:, 0], y=df.iloc[:, 6], hue = df.iloc[:, 3]) # Change positional index to match the location of variables.
  # # ax.set_xlabel('X-axis values')
  # # ax.set_ylabel('Y-axis values')
  # st.pyplot(fig)
  # 
  st.write("##")
  st.write("##")
  st.subheader("""Correlation heatmap""")
  fig, ax = plt.subplots()
  sns.heatmap(df.corr(), ax = ax)
  st.write(fig, use_container_width=False)
else:
  st.warning(':point_left:  Awaiting user\'s input file :exclamation:')

