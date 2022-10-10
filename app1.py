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
else:
  st.warning(':point_left:  Awaiting user\'s input file :exclamation:')

# st.write("##")
# st.write("##")
# st.write("---")
#     
# @st.cache(persist=True)
# def split(df):
#     y = df.target
#     x = df.drop(columns=["target"])
#     x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3, random_state=0)
#     return x_train, x_test, y_train, y_test
#   
#     x_train, x_test, y_train, y_test = split(df)
#     
# def plot_metrics(metrics_list):
#   if "Confusion Matrix" in metrics_list:
#       st.subheader("Confusion Matrix")
#       plot_confusion_matrix(model, x_test, y_test, display_labels=   class_names)
#       st.pyplot()
#   if "ROC Curve" in metrics_list:
#       st.subheader("ROC Curve")
#       plot_roc_curve(model, x_test, y_test)
#       st.pyplot()
#   if "Precision-Recall Curve" in metrics_list:
#       st.subheader("Precision-Recall Curve")
#       plot_precision_recall_curve(model, x_test, y_test)
#       st.pyplot()
# class_names = ["Benign", "Malignant"]
#  
# st.sidebar.subheader("Choose classifier")
# classifier = st.sidebar.selectbox("Classifier", ("Support Vector Machine (SVM)", "Logistic Regression", "Random Forest"))
# 
# if classifier == "Support Vector Machine (SVM)":
#     st.sidebar.subheader("Hyperparameters")
#     C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key="C")
#     kernel = st.sidebar.radio("Kernel", ("rbf", "linear"), key="kernel") 
#     gamma = st.sidebar.radio("Gamma (Kernal coefficient", ("scale", "auto"), key="gamma")
#     metrics = st.sidebar.multiselect("What metrics to plot?", ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve"))
#     
#     if st.sidebar.button("Classify", key="classify"):
#       st.subheader("Support Vector Machine (SVM) results")
#       df = load()
#       x_train, x_test, y_train, y_test = split(df)
#       model = SVC(C=C, kernel=kernel, gamma=gamma)
#       model.fit(x_train, y_train)
#       accuracy = model.score(x_test, y_test)
#       y_pred = model.predict(x_test)
#       st.write("Accuracy: ", accuracy.round(2))
#       st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
#       st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2)) 
#       plot_metrics(metrics)
# 
# if classifier == "Logistic Regression":
#     st.sidebar.subheader("Hyperparameters")
#     C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key="C_LR")
#     max_iter = st.sidebar.slider("Maximum iterations", 100, 500, key="max_iter")
#     metrics = st.sidebar.multiselect("What metrics to plot?", ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve"))
#     
#     if st.sidebar.button("Classify", key="classify"):
#         st.subheader("Logistic Regression Results")
#         model = LogisticRegression(C=C, max_iter=max_iter)
#         model.fit(x_train, y_train)
#         accuracy = model.score(x_test, y_test)
#         y_pred = model.predict(x_test)
#         
#         st.write("Accuracy: ", accuracy.round(2))
#         st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
#         st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
#         plot_metrics(metrics)
