import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

def load_data():
	df = pd.read_csv('https://student-datasets-bucket.s3.ap-south-1.amazonaws.com/whitehat-ds-datasets/adult.csv', header=None)
	df.head()
	column_name =['age', 'workclass', 'fnlwgt', 'education', 'education-years', 'marital-status', 'occupation', 'relationship', 'race','gender','capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']
	for i in range(df.shape[1]):
	  df.rename(columns={i:column_name[i]},inplace=True)
	df.head()
	df['native-country'] = df['native-country'].replace(' ?',np.nan)
	df['workclass'] = df['workclass'].replace(' ?',np.nan)
	df['occupation'] = df['occupation'].replace(' ?',np.nan)
	df.dropna(inplace=True)
	df.drop(columns='fnlwgt',axis=1,inplace=True)
	return df

df = load_data()
st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(layout='centered')
st.header('Census Data Visualization')
st.subheader('Different graphs and tables will help you understand this data')
if st.checkbox('View Raw Data'):
	st.table(df.head())
	endt =  st.checkbox('Show Entire Data')
	if endt:
		st.dataframe(df)
if st.checkbox('View Dataframe Info'):
	st.table(df.info())
if st.checkbox('View Dataframe Description'):
	st.table(df.describe())
if st.checkbox('View Dataframe Size'):
	st.write(df.shape[0], ' rows\n', df.shape[1], ' columns')
if st.checkbox('Plot income group'):
	plt.figure(figsize=(5, 5))
	plt.pie(df['income'].value_counts(), autopct='%.2f', labels=df['income'].value_counts().keys(), shadow=True,wedgeprops = {'linewidth': 3}, textprops={'color':'black'})
	st.pyplot()
if st.checkbox('Plot hours-per-week feature for different income groups'):
	plt.figure(figsize=(5, 5))
	sns.boxplot(df['hours-per-week'], df['income'])
	st.pyplot()
if st.checkbox('Plot hours-per-week feature for different gender groups'):
	plt.figure(figsize=(5, 5))
	sns.boxplot(df['hours-per-week'], df['gender'])
	st.pyplot()
if st.checkbox('Correlate between workclass feature for different income groups'):
	plt.figure(figsize=(10, 5))
	wc = df['workclass']
	inc = df['income']
	mapd = {}
	for i in range(0,len(wc.unique())):
		mapd[wc.unique()[i]] = i
	wc = wc.map(mapd)
	mapd2 = {}
	for i in range(0,len(inc.unique())):
		mapd2[inc.unique()[i]] = i
	inc = inc.map(mapd)
	wcinc = pd.DataFrame({'workclass':wc, 'income':inc})
	sns.heatmap(wcinc.corr())
	st.pyplot()