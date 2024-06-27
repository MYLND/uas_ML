
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix

# Set seaborn style
sns.set(style='dark')

# Load data
data = pd.read_csv('Phising_dataset.csv')

# Page title
st.title("Phishing Website Detection Dashboard")

# Display dataset
phisingweb = data('Phising_dataset.csv')
st.markdown('<h5 style="text-align:justify">Menampilkan Dataset Web Phishing</h5>', unsafe_allow_html=True)
with st.expander('Phising_dataset', expanded=False):
    st.dataframe(phisingweb)
st.dataframe(data.head())

# Phishing count pie chart
st.header("Phishing URL Count")
phishing_counts = data['Result'].value_counts()
fig1, ax1 = plt.subplots()
phishing_counts.plot(kind='pie', autopct='%1.2f%%', ax=ax1, startangle=90, colors=["#ff9999","#66b3ff"])
ax1.set_ylabel('')
plt.title("Phishing URL Distribution")
st.pyplot(fig1)

# Confusion matrix
st.header("Confusion Matrix for Naive Bayes Model")

# Given confusion matrix values (as per previous calculations)
cm = np.array([[162*0.46, 162*(1-0.46)],
               [118*(1-0.99), 118*0.99]])

fig2, ax2 = plt.subplots()
sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", xticklabels=["Predicted -1", "Predicted 1"], yticklabels=["True -1", "True 1"], ax=ax2)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
st.pyplot(fig2)
