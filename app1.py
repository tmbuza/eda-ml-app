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
import plotly.express as px

#--------------------------------
# PAGE CONFIG
#--------------------------------
st.set_page_config(
     page_title='ML WebApps',
     page_icon=':bar_chart:',
     layout='wide',
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
      
      
    with st.container():
      st.success(
      """ 
      # :shower: Data Exploration App
      """)
      
      col1, col2 = st.columns(2)
      with col1:
        st.image(cover, width=450)        
      with col2:
        st.markdown(
          """
          ### App Functionalities
          - This App helps you to create a tidy dataframe for downstream machine learning problems.
            - Reads the `data` uploaded by user and automatically starts exploration.
            - Displays the `head` and `tail` of the uploaded data (input dataframe).
            - Shows the dataframe `dimension`, `variable names` and `missing values`.
            - Perform `descriptive` statistical analysis of the numerical variables.
            - Plots a `correlation` heatmap.
            - Create Plotly interactive plots
          """)

        st.write("""### Testing the App""")
        st.markdown(
          """
          - You may start by testing the app using a demo data by clicking on the `Run a Partial Demo` button, otherwise:
          - Upload a tidy input file using the user\'s widget on the left sidebar.
          """)
        if st.button('Run a Partial Demo'):
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
          
if __name__ == '__main__':
  main()
  
st.write("---")

st.header("Start Exploring Your Data")    
st.write(":point_left:  Use the widgets on the left to get started.") 

# Functions for each of the pages
def home(uploaded_file):
  if uploaded_file:
    st.subheader('User data is uploaded and ready!.')
  # else:
  #   st.warning('Awaiting user\'s input file :exclamation:')
  # 
def input_dataframe():
  st.subheader("""Dataframe Structure""")
  st.write("Head", df.head())
  st.write("Tail", df.tail())

  st.subheader("""Dataframe dimension""")
  st.markdown("> Note that 0 = rows, and 1 = columns")
  st.dataframe(df.shape) 

def col_names():  
  st.subheader("""Variable Names""")
  st.dataframe(pd.DataFrame({'Variable': df.columns}))
  
def data_stats():
  st.header("""Basic Statistics""")
  st.subheader("""Descriptive statistics""")
  st.dataframe(df.describe())

def missing_data():
  st.subheader("""Missing Data""")
  missing_count = df.isnull().sum()
  value_count = df.isnull().count() #denominator
  missing_percentage = round(missing_count/value_count*100, 2)
  missing_df = pd.DataFrame({'Count': missing_count, 'Missing (%)': missing_percentage})
  st.dataframe(missing_df)

def heatmap():
  st.subheader("""Correlation Heatmap""")
  fig, ax = plt.subplots()
  sns.heatmap(df.corr(), ax = ax)
  st.write(fig, use_container_width=False)

def interactive_plot():
    col1, col2 = st.columns(2)
    
    x_axis_val = col1.selectbox('Select data for the X-axis', options=df.columns)
    y_axis_val = col2.selectbox('Select data for the Y-axis', options=df.columns)

    plot = px.scatter(df, x=x_axis_val, y=y_axis_val)
    st.plotly_chart(plot, use_container_width=True)

# Sidebar setup
st.sidebar.title('User\'s Input Widget')
upload_file = st.sidebar.file_uploader("Please choose a tidy CSV file", type=["csv"])

#Sidebar navigation
st.sidebar.title('Exploration Navigation')

options = st.sidebar.radio('Please Select what to Display:', options = [
  'Home', 
  'Dataframe Structure', 
  'Variable Names', 
  'Missing Data',
  'Descriptive Statistics', 
  'Correlation Heatmap',
  'Interactive Scatter Plots'])

# Check if file has been uploaded
if upload_file:
  df = pd.read_csv(upload_file)
else:
  st.warning('Awaiting user\'s input file :exclamation:')
 
# Navigation options
if options == 'Home':
    home(upload_file)
elif options == 'Dataframe Structure':
    input_dataframe()
elif options == 'Variable Names':
    col_names()
elif options == 'Descriptive Statistics':
    data_stats()
elif options == 'Missing Data':
    missing_data()
elif options == 'Correlation Heatmap':
    heatmap()
elif options == 'Interactive Scatter Plots':
    interactive_plot()    


