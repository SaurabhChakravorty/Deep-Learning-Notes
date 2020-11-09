import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from seglearn.transform import Segment

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from keras import backend as K

import holidays
de_holidays = holidays.DE()

RANDOM_SEED = 42

data_path = '../data/'

def train_test_valid_split(df, window_size, feature_len, 
						   split_pct_1=0.3, split_pct_2=0.33, 
						   test_set=True):
	"""Splits data into training, validation and test sets. 
	If you do not want a test set, set the `test_set` param
	to False. """
	X_train, X_valid, y_train, y_valid = train_test_split(df.iloc[:,:(window_size*feature_len)], 
														  df.iloc[:,(window_size*feature_len):-1], 
														  test_size=split_pct_1, shuffle=True, 
														  random_state=RANDOM_SEED)
	# print(y_valid.shape, type(y_valid),'\n' , X_train.shape, type(X_train))

	if test_set:
		X_valid, X_test, y_valid, y_test = train_test_split(X_valid, y_valid, 
															test_size=split_pct_2, shuffle=True, 
															random_state=RANDOM_SEED)
		
		return X_train, X_valid, X_test, y_train.iloc[:,0].values, y_valid.iloc[:,0].values, y_test.iloc[:,0].values
	
	return X_train, X_valid, y_train.iloc[:,0].values, y_valid.iloc[:,0].values

def calc_reg_metrics(y_true, y_pred):
	"""Calculates a set of regression 
	metrics"""
	mse = mean_squared_error(y_true, y_pred)
	rmse = np.sqrt(mse)
	mae = mean_absolute_error(y_true, y_pred)
	try:
		mape = mean_absolute_percentage_error(y_true, y_pred)
	except:
		pass
	r2 = r2_score(y_true, y_pred)
	results = pd.DataFrame([mse, rmse, mae, r2], 
						index=['MSE', 'RMSE', 'MAE', 'R2'], 
						columns=['value'])
	return results

def create_column_features(features, window_size):
	"""Create column names from list 
	of features and window size"""
	columns = []
	for i in list(range(window_size)) + ['y']:
		for f in features:
			columns.append(f+'_'+str(i))
	return columns 

def create_features(temp, features: list, ohlc: bool=True):
	"""Creates features based on list. """
	if ohlc:
		y = temp.px.close.values
	else:
		y = temp.px.values
	feature_list = []
	if 'weekday' in features:
		weekday = np.array(temp.index.dayofweek) 
		feature_list.append(weekday)
	if 'weekday_sin' in features:
		weekday_sin = np.sin(2*np.pi*temp.index.dayofweek/6)
		feature_list.append(weekday_sin)
	if 'weekday_cos' in features:
		weekday_cos = np.cos(2*np.pi*temp.index.dayofweek/6)
		feature_list.append(weekday_cos)
	if 'run_hour' in features:
		feature_list.append(temp.hour)
	if 'hours_to_4' in features:
		# hour = temp.index.hour
		hours_to_4 = np.array([40-hour if hour>16 else 16-hour for hour in temp.index.hour])/23
		feature_list.append(hours_to_4)
	if 'n_prev_hour_contracts' in features:
		feature_list.append(temp.n_prev_hour_contracts/41)
	if 'hour' in features:
		hour = np.array(temp.index.hour)
		feature_list.append(hour)
	if 'hour_sin' in features:
		hour_sin = np.sin(2*np.pi*temp.index.hour/23)
		feature_list.append(hour_sin)
		# 16 - temp.index.hour
	if 'hour_cos' in features:
		hour_cos = np.cos(2*np.pi*temp.index.hour/23)
		feature_list.append(hour_cos)
	if 'air_temp' in features:
		feature_list.append(temp.air_temp)
	if 'rel_humidity' in features:
		feature_list.append(temp.rel_humidity)
	if 'wind_speed' in features:
		feature_list.append(temp.wind_speed)
	if 'wind_dir' in features:
		feature_list.append(temp.wind_dir)
	if 'holidays' in features:
		holidays = np.array([x in de_holidays for x in temp.index.strftime("%Y-%m-%d")])
		feature_list.append(holidays)
	if 'qty_open' in features:
		qty_open = np.array(temp.qty.open.values)
		feature_list.append(qty_open)
	if 'qty_high' in features:
		qty_high = np.array(temp.qty.high.values)
		feature_list.append(qty_high)
	if 'qty_low' in features:
		qty_low = np.array(temp.qty.low.values)
		feature_list.append(qty_low)
	if 'qty_close' in features:
		qty_close = np.array(temp.qty.close.values)
		feature_list.append(qty_close)
	if 'qty_var' in features:
		try:
			qty_var = np.array(temp.qty['var'].values)
		except:
			qty_var = np.array(temp.qty.qty.values)
		feature_list.append(qty_var)
	if 'qty_sum' in features:
		try:
			qty_sum = np.array(temp.qty['sum'].values)
		except:
			qty_sum = np.array(temp.qty.qty.values)
		feature_list.append(qty_sum)
	if 'act_px_open' in features:
		act_px_open = np.array(temp.act_px.open.values)
		feature_list.append(act_px_open)
	if 'act_px_high' in features:
		act_px_high = np.array(temp.act_px.high.values)
		feature_list.append(act_px_high)
	if 'act_px_low' in features:
		act_px_low = np.array(temp.act_px.low.values)
		feature_list.append(act_px_low)
	if 'act_px_close' in features:
		act_px_close = np.array(temp.act_px.close.values)
		feature_list.append(act_px_close)
	if 'px_open' in features:
		px_open = np.array(temp.px.open.values)
		feature_list.append(px_open)
	if 'px_high' in features:
		px_high = np.array(temp.px.high.values)
		feature_list.append(px_high)
	if 'px_low' in features:
		px_low = np.array(temp.px.low.values)
		feature_list.append(px_low)
	if 'px_var' in features:
		px_var = np.array(temp.px['var'].values)
		feature_list.append(px_var)
	if 'act_px_absdif' in features:
		act_px_absdif = np.array(temp.act_px_absdif.values)
		feature_list.append(act_px_absdif)
	if 'px_absdif' in features:
		px_absdif = np.array(temp.px_absdif.values)
		feature_list.append(px_absdif)
		
	return np.stack([y, *feature_list], axis=1), y

def create_rolling_windows(resampled_df: pd.DataFrame, window_size: int,
						   features: list, save_to_pickle: bool=True, 
						   ohlc: bool=True) -> pd.DataFrame:
	"""Creates rolling windows from the data. You need to specify
	a window size and a list of feature names you have."""
	if ohlc:
		contracts = resampled_df['contractId']['contractId'].value_counts()\
				[resampled_df['contractId']['contractId'].value_counts() > window_size].index
	else:
		contracts = resampled_df['contractId'].value_counts()\
					[resampled_df['contractId'].value_counts() > window_size].index
	columns = create_column_features(features, window_size)
	segmenter = Segment(width=window_size+1, step=1)
	forecast_df = pd.DataFrame()
	for c in contracts: 
		if ohlc:
			temp = resampled_df[resampled_df['contractId']['contractId']==c]
			save_str = 'ohlc'
			date = '27102020'
		else:
			temp = resampled_df[resampled_df['contractId']==c]
			save_str = 'last'
			date = '25102020'
		X, y = create_features(temp, features, ohlc)
		X_train, y_train, _ = segmenter.fit_transform([X], [y])
		assert X_train.shape[0] == len(temp) - window_size
		temp_rolling = pd.DataFrame(X_train.reshape(X_train.shape[0], -1), columns=columns)
		temp_rolling['contractId'] = c
		forecast_df = pd.concat([forecast_df, temp_rolling])
	forecast_df.reset_index(drop=True, inplace=True)
	if save_to_pickle:
		forecast_df.to_pickle(data_path+f'rolling_{window_size}_{save_str}_{date}.pkl', compression='zip')
	return forecast_df

def bin_ohlcv(df, contractId, binning_size='H'):
	df_cid = df[df.contractId == contractId]
	# resample for a binsize and the ohlc the result; and volume too.
	data = df_cid[['px']].resample(binning_size).ohlc().px
	data['volsum'] = df_cid[['qty']].resample(binning_size).sum()
	return data

def plot_ohlcv(df, contractId, binning_size='H'):
	data = bin_ohlcv(df, contractId, binning_size)
	fig = make_subplots(specs=[[{"secondary_y": True}]])
	trace1 = go.Candlestick(x=data.index,
					 open=data['open'],
					 high=data['high'],
					 low=data['low'],
					 close=data['close'], 
					 name=contractId)
	trace2 = go.Bar(x=data.index, 
					y=data['volsum'], 
					name='Volume', 
					opacity=.5,
					marker={'color': 'blue'})
	fig.add_trace(trace1)
	fig.add_trace(trace2, secondary_y=True)
	fig.update_layout(title=f'OHLCV for {contractId}')
	fig.update_layout(xaxis_rangeslider_visible=False)
	fig.show()

def remove_outliers(df, method, thresh, 
					window_size):
	"""Removes outliers from data based on
	Variance or Std. Dev."""
	cols = [f't_{i}' for i in range(window_size)] + ['t_y']
	if method=='stddev':
		vals = df[cols].std(axis=1)
	elif method=='var':
		vals = df[cols].var(axis=1)
	elif method=='zscore':
		vals = df[cols].var(axis=1)
		z = zscore(vals)

	else:
		raise ValueError('Outlier Removal Method \
			is not supported. Try `stddev` or `var`.')
	vals = vals[np.abs(vals)<thresh]
	print(f'Dropped {len(df)-len(vals)} rows as outliers, \
		\nkeeping {np.round((len(vals)/len(df))*100, 2)}% of rows')
	return df.loc[vals.index]


def coeff_determination(y_true, y_pred):
	SS_res =  K.sum(K.square(y_true-y_pred))
	SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
	return (1 - SS_res/(SS_tot + K.epsilon()))