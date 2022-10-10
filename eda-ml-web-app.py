from numpy.core.numeric import True_
from sklearn import metrics
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score

# Page configuration
st.set_page_config(
     page_title='ML WebApps',
     page_icon=':bar_chart:',
     layout='centered',
     initial_sidebar_state='expanded')

def main():
    st.title(" Machine Learning Web Applications using Python and Streamlit Library")
    st.header("Simple Streamlit WebApp")
    st.warning(""":warning: **If new to python check this [link](https://www.python.org/about/gettingstarted/)**""")

    st.sidebar.title("This is the sidebar")
    st.sidebar.markdown("Let’s start with binary classification!!")
if __name__ == '__main__':
  main()

@st.cache(persist= True)
def load():
    data= pd.read_csv("data/breast_cancer.csv")
    label= LabelEncoder()
    for i in data.columns:
        data[i] = label.fit_transform(data[i])
    return data

if st.sidebar.checkbox("Display test data", False):
    st.subheader("Test dataset")
    df = load()
    st.write(df)
    
st.write("---")
st.write("##")

st.write("""### User input widget""")
st.markdown(
  """ """)
uploaded_file = st.file_uploader("Please choose a CSV file", type=["csv"])  
if uploaded_file is not None:    
  def load_data():
      a = pd.read_csv(uploaded_file)
      return a
  df = load_data()
  st.header("""Data Exploration""")
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

  st.subheader("""Correlation heatmap""")    
  fig, ax = plt.subplots()
  sns.heatmap(df.corr(), ax = ax)
  st.write(fig, use_container_width=False) 
else:
  st.warning(':exclamation: Awaiting user\'s input file')


st.write("##")
st.write("---")
    
@st.cache(persist=True)
def split(df):
    y = df.target
    x = df.drop(columns=["target"])
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3, random_state=0)
    return x_train, x_test, y_train, y_test
  
    x_train, x_test, y_train, y_test = split(df)
    
def plot_metrics(metrics_list):
  if "Confusion Matrix" in metrics_list:
      st.subheader("Confusion Matrix")
      plot_confusion_matrix(model, x_test, y_test, display_labels=   class_names)
      st.pyplot()
  if "ROC Curve" in metrics_list:
      st.subheader("ROC Curve")
      plot_roc_curve(model, x_test, y_test)
      st.pyplot()
  if "Precision-Recall Curve" in metrics_list:
      st.subheader("Precision-Recall Curve")
      plot_precision_recall_curve(model, x_test, y_test)
      st.pyplot()
class_names = ["edible", "poisnous"]
 
st.sidebar.subheader("Choose classifier")
classifier = st.sidebar.selectbox("Classifier", ("Support Vector Machine (SVM)", "Logistic Regression", "Random Forest"))

if classifier == "Support Vector Machine (SVM)":
    st.sidebar.subheader("Hyperparameters")
    C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key="C")
    kernel = st.sidebar.radio("Kernel", ("rbf", "linear"), key="kernel") 
    gamma = st.sidebar.radio("Gamma (Kernal coefficient", ("scale", "auto"), key="gamma")
    metrics = st.sidebar.multiselect("What metrics to plot?", ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve"))
    
    if st.sidebar.button("Classify", key="classify"):
      st.subheader("Support Vector Machine (SVM) results")
      df = load()
      x_train, x_test, y_train, y_test = split(df)
      model = SVC(C=C, kernel=kernel, gamma=gamma)
      model.fit(x_train, y_train)
      accuracy = model.score(x_test, y_test)
      y_pred = model.predict(x_test)
      st.write("Accuracy: ", accuracy.round(2))
      st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
      st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2)) 
      plot_metrics(metrics)

if classifier == "Logistic Regression":
    st.sidebar.subheader("Hyperparameters")
    C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key="C_LR")
    max_iter = st.sidebar.slider("Maximum iterations", 100, 500, key="max_iter")
    metrics = st.sidebar.multiselect("What metrics to plot?", ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve"))
    
    if st.sidebar.button("Classify", key="classify"):
        st.subheader("Logistic Regression Results")
        model = LogisticRegression(C=C, max_iter=max_iter)
        model.fit(x_train, y_train)
        accuracy = model.score(x_test, y_test)
        y_pred = model.predict(x_test)
        
        st.write("Accuracy: ", accuracy.round(2))
        st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
        st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
        plot_metrics(metrics)
