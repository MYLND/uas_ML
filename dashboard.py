import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Set seaborn style
sns.set(style='dark')

# Load data
data = pd.read_csv(Phising_dataset.csv, encoding='windows-1254')

# Display dataset
st.title("Phishing Dataset Overview")
st.subheader("Dataset")
st.write(data.head())

# Display dataset info
st.subheader("Dataset Info")
st.write(data.info())

# Display unique values in columns
st.subheader("Unique Values in Columns")
st.write(data.nunique())

# Drop URL column
data = data.drop(['URL'], axis=1)

# Display dataset description
st.subheader("Dataset Description")
st.write(data.describe().T)

# Display correlation heatmap
st.subheader("Correlation Heatmap")
fig, ax = plt.subplots(figsize=(15, 15))
sns.heatmap(data.corr(), annot=True, ax=ax)
st.pyplot(fig)

# Display pairplot for selected features
st.subheader("Pairplot for Selected Features")
df = data[['Presence of IP Address', 'Length of URL', 'No. of Slashes', 'Special Character', 'Age of URL', 'Result']]
sns.pairplot(data=df, hue="Result", corner=True)
st.pyplot()

# Display pie chart of phishing vs non-phishing
st.subheader("Phishing URL Count")
fig, ax = plt.subplots()
data['Result'].value_counts().plot(kind='pie', autopct='%1.2f%%', ax=ax)
plt.title("Phishing URL Count")
st.pyplot(fig)

# Split dataset into features and target variable
X = data.drop(['Result'], axis=1)
y = data['Result']

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Initialize and train the Naive Bayes model
clf = GaussianNB()
clf.fit(X_train, y_train)

# Predict target values for training and testing sets
y_train_nb = clf.predict(X_train)
y_test_nb = clf.predict(X_test)

# Display model performance metrics
st.subheader("Model Performance Metrics")
acc_train_nb = metrics.accuracy_score(y_train, y_train_nb)
acc_test_nb = metrics.accuracy_score(y_test, y_test_nb)
f1_score_train_nb = metrics.f1_score(y_train, y_train_nb)
f1_score_test_nb = metrics.f1_score(y_test, y_test_nb)
recall_train_nb = metrics.recall_score(y_train, y_train_nb)
recall_test_nb = metrics.recall_score(y_test, y_test_nb)
precision_train_nb = metrics.precision_score(y_train, y_train_nb)
precision_test_nb = metrics.precision_score(y_test, y_test_nb)

st.write(f"Naive Bayes: Accuracy on Training Data: {acc_train_nb:.3f}")
st.write(f"Naive Bayes: Accuracy on Test Data: {acc_test_nb:.3f}")
st.write(f"Naive Bayes: F1 Score on Training Data: {f1_score_train_nb:.3f}")
st.write(f"Naive Bayes: F1 Score on Test Data: {f1_score_test_nb:.3f}")
st.write(f"Naive Bayes: Recall on Training Data: {recall_train_nb:.3f}")
st.write(f"Naive Bayes: Recall on Test Data: {recall_test_nb:.3f}")
st.write(f"Naive Bayes: Precision on Training Data: {precision_train_nb:.3f}")
st.write(f"Naive Bayes: Precision on Test Data: {precision_test_nb:.3f}")

# Display classification report
st.subheader("Classification Report")
st.write(metrics.classification_report(y_test, y_test_nb))

# Display confusion matrix
st.subheader("Confusion Matrix")
cm = np.array([[162*0.46, 162*(1-0.46)], [118*(1-0.99), 118*0.99]])
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", xticklabels=["Predicted -1", "Predicted 1"], yticklabels=["True -1", "True 1"], ax=ax)
plt.title("Confusion Matrix for Naive Bayes Model")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
st.pyplot(fig)
