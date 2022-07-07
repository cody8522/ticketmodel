import pandas as pd
import numpy as np
import streamlit as st
import random
import pickle
import zipfile
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
import warnings
from os import listdir
from os.path import isfile, join
import ast
from sklearn import preprocessing
warnings.filterwarnings('ignore')
import random
import json
import datetime
from datetime import datetime
from sklearn import preprocessing
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.ensemble import RandomForestRegressor
import pyarrow.parquet as pq


print('started df formation')


extradroplst = ['Section','Artist Name', 'Name', 'City', 'State','DayOfWeek','Month']
chartcols = ['sp_followers', 'sp_popularity', 'sp_followers_to_listeners_ratio', 'sp_monthly_listeners',
	 'sp_playlist_total_reach','cm_artist_rank','cm_artist_score','facebook_followers','ins_followers']


@st.experimental_memo(suppress_st_warning = True)
def load_df1():
	table = pq.read_table("model_startup.parquet")
	df = table.to_pandas()
	return df
df = load_df1()


artist_list = ['1', '2', '3']
venue_list = ['1','2','3']
city_list = ['1','2','3']



@st.experimental_memo(suppress_st_warning = True)
def prevshows(artistvar,venuevar,sectionvar,eventfull,mydisplay=True):
	chartcols = ['sp_followers',
 'sp_popularity',
 'sp_followers_to_listeners_ratio',
 'sp_monthly_listeners',
 'sp_playlist_total_reach',
 'cm_artist_rank','cm_artist_score','facebook_followers','ins_followers']
	
	event = eventfull.copy()        	
	event = event.merge(pd.read_csv('Venue Information.csv').rename(columns={'Venue':'Name'})[['Name','Adjusted Capacity']],on='Name',how='left')

	chart = pd.read_csv('chartmetric - chartmetric.csv')
	chart['name'] = list(map(lambda x: x.strip().title(), chart['name']))
	chart = chart[['name']+chartcols].dropna().rename(columns={'name':'Artist Name'})
	
	event = event.merge(chart,on='Artist Name',how='left')
	
	
	for c in ['Artist Name','Name','City','State','DayOfWeek','Month','Section']:
		
		try:
			event[c] = pickle.load(open('encode_dict.pkl','rb'))[c][event[c].iloc[0]]
		except:
			if mydisplay:
				print(c)
			event[c] = event['Artist Name'].iloc[0]
			if mydisplay:
				print('Imputed ' + c + ' To Artist Average')
		

	event = event.rename(columns={'Name':'Venue'})

	eventmerge = event.copy()
	mldf_list = ['Artist Name', 'Venue', 'Section', 'Artist-Venue', 'Artist-Section', 'Venue-Section', 'Artist-Venue-Section', 'DayOfWeek', 'Month',
	 'Adjusted Capacity', 'City', 'State', 'Cons1', 'Cons2', 'sp_followers', 'sp_popularity',
	 'sp_followers_to_listeners_ratio', 'sp_monthly_listeners', 'sp_playlist_total_reach', 'cm_artist_rank', 'cm_artist_score', 'facebook_followers', 'ins_followers']

	eventmerge = eventmerge.merge(pickle.load(open('Artist-Section.pkl','rb')),how='left',on=['Artist Name','Section'])
	eventmerge = eventmerge.merge(pickle.load(open('Artist-Venue.pkl','rb')),how='left',on=['Artist Name','Venue'])
	eventmerge = eventmerge.merge(pickle.load(open('Venue-Section.pkl','rb')),how='left',on=['Venue','Section'])
	eventmerge = eventmerge.merge(pickle.load(open('Artist-Venue-Section.pkl','rb')),how='left',on=['Artist Name','Venue','Section'])
	eventmerge = eventmerge[mldf_list]
	
	varlst = ['All','Artist-Venue','Artist-Section','Venue-Section','Artist-Venue-Section']

	predstree = {}
	predslinear = {}
	
	for i in varlst:
		if i!='All' and np.isnan(eventmerge.iloc[0][i]):
			pass
		else:
			model = pickle.load(open('model_dict.sav','rb'))[i]
			droplst = [q for q in varlst if q!=i]
			droplst = [q for q in droplst if q!='All']
			if i!='Artist-Venue-Section':
				test = eventmerge.drop(droplst,axis=1)
			else:
				test = eventmerge.copy()
			predstree[i] = model.predict(test.values)[0]
			

	droplst = [i for i in droplst if i!='All'] + [i if i!='Name' else 'Venue' for i in extradroplst]
	droplst = [q for q in varlst if q!='All']
	test = eventmerge[['Adjusted Capacity',
 'Cons1',
 'Cons2',
 'sp_followers',
 'sp_popularity',
 'sp_followers_to_listeners_ratio',
 'sp_monthly_listeners',
 'sp_playlist_total_reach',
 'cm_artist_rank',
 'cm_artist_score',
 'facebook_followers',
 'ins_followers']]

	
	import statistics as st
	
	findf = pd.concat([pd.DataFrame(predstree,index=range(1)).T.rename(columns={0:'Tree Based'}),pd.DataFrame(predslinear,index=range(1)).T.rename(columns={0: 'Linear'})],axis=1)
	if mydisplay:
		display(findf)
	try:
		pred = findf.loc['Artist-Venue-Section'].mean()
		print('Artist-Venue-Section Model')
		return [pred, [pred - pred*0.15, pred + pred*0.15],[pred - pred*0.35, pred + pred*0.35],[pred - pred*0.50, pred + pred*0.50]]
	except:
		try:
			pred = st.mean(list(findf.drop(['All'])['Tree Based']))
			print('Two-Combo Model')
			return [pred, [pred - pred*0.20, pred + pred*0.20],[pred - pred*0.40, pred + pred*0.40],[pred - pred*0.625, pred + pred*0.625]]
		except:
			pred = findf.loc['All'].mean()
			print('Individual Model')
			return [pred, [pred - pred*0.275, pred + pred*0.275],[pred - pred*0.45, pred + pred*0.45],[pred - pred*0.65, pred + pred*0.65]]

@st.experimental_memo(suppress_st_warning = True)       
def showmodel(df, venue, artist, month, DayOfWeek, city, state, cons1, cons2, displaybool=True):
	venuevar = venue
	artistvar = artist
	mydict = {}
	for i in list(df[df['Name']==venuevar]['Section'].value_counts().keys()):
		eventfull = pd.DataFrame({'Artist Name': artist,
		 'Section': i,#'Balcony Center',
		 'Month': month,
		 'DayOfWeek': DayOfWeek,
		 'Name': venuevar,
		 'City': city,
		 'State': state,
		 'Cons1': cons1,
		 'Cons2': cons2},index=range(1))

		artistvar = eventfull['Artist Name'].iloc[0]
		venuevar = eventfull['Name'].iloc[0]
		sectionvar = eventfull['Section'].iloc[0]
		a = prevshows(artistvar,venuevar,sectionvar,eventfull,mydisplay=displaybool)
		mydict[i] = a
		
	findf = pd.DataFrame(mydict,index=range(4)).T.rename(columns={0:artistvar+'-'+venuevar+' Prediction',1:'50% Confidence Interval', 2:'75% Confidence Interval',3:'90% Confidence Interval'})
	for c in findf.columns:
		try:
			findf[c] = [round(i,2) for i in list(findf[c])]
		except:
			findf[c] = [[round(i[0],2),round(i[1],2)] for i in findf[c]]
	
	return findf
	
print("2")
st.write("""
# TexTickets Fair Market Value App

### Select your parameters for a FMV estimation.

""")
def user_input_features():
	with st.sidebar:
		st.header('User Input Parameters')
		with st.sidebar.form("form"):
			ArtistName = st.selectbox('Artist Name', artist_list)
			Month = st.selectbox('Month', ('1','2','3','4','5','6','7','8','9','10','11','12'))
			DayOfWeek = st.selectbox('DayOfWeek', ('1','2','3','4','5','6','7'))
			State = st.selectbox('State', ('AL', 'AK', 'AZ', ' AR', 'CA', 'CO', 'CT', 'DE', 'DC', 'FL', 'GA', 'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI', 'MN',
				'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY'))
			City = st.selectbox('City', city_list)
			Name = st.selectbox('Venue', venue_list)
			Section = st.selectbox('Section', ('All', 'None'))
			Cons1 = st.selectbox('Cons1', ('0', '1'))
			Cons2 = st.selectbox('Cons2', ('0', '1'))
			data = {'Artist Name' : ArtistName,
			'Section' : Section,
			'Month' : Month,
			'DayOfWeek' : DayOfWeek,
			'Name' : Name,
			'City': City,
			'State' : State,
			'Cons1' : Cons1,
			'Cons2' : Cons2}
			button_check = st.form_submit_button("Submit")
			features = pd.DataFrame(data, index = [0])
			return features
stdf = user_input_features()
st.subheader('User Selections')
st.write(stdf)

###### Model Run

artistvar = stdf['Artist Name'].iloc[0]
venuevar = stdf['Name'].iloc[0]
sectionvar = stdf['Section'].iloc[0]
cityvar = stdf['City'].iloc[0]
statevar = stdf['State'].iloc[0]
monthvar = stdf['Month'].iloc[0]
dayvar = stdf['DayOfWeek'].iloc[0]
cons1var = stdf['Cons1'].iloc[0]
cons2var = stdf['Cons2'].iloc[0]


a = showmodel(df=df, venue=venuevar, artist=artistvar, month=monthvar, DayOfWeek=dayvar, city=cityvar, state=statevar, cons1=cons1var, cons2=cons2var,displaybool=False)
st.write(a)



