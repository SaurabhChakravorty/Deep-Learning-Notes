import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from seglearn.transform import Segment
import xgboost as xgb
import glob
import lib
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Activation, TimeDistributed, Flatten

import holidays
de_holidays = holidays.DE()

RANDOM_STATE = 42
data_path = '../data/'

class Forecasting:
	"""One object to rule them all. Right now, 
	it only support closing price forecasting
	based on closing price info. """

	def __init__(self, df, window_size, features, 
		regressor:str, params: dict=None, split_pct_1=0.2, 
		split_pct_2=0.5, test_set=False, ohlc=True, 
		recalculate=False, grid_search=False, 
		save_to_pickle: bool=True, custom_model=None, 
		outlier_removal: str=None, outlier_thresh=None):
		"""Here is how to specify the regressors; keep
		adding a standardised name below if you add a 
		new regressor. Also update the `model_selector`
		funtion below. 

		`randFor`: RandomForestRegressor
		`gradBoost`: GradientBoostingRegressor
		`xgBoost`: XGBoost
		`lstm`: LSTM
		`custom`: Custom Model is passed

		"""
		self.df = df
		self.WINDOW_SIZE = window_size
		self.features = features
		self.test_set = test_set
		self.regressor = regressor
		self.params = params
		self.split_pct_1 = split_pct_1
		self.split_pct_2 = split_pct_2
		self.ohlc = ohlc
		self.recalculate = recalculate
		self.grid_search = grid_search
		self.save_to_pickle = save_to_pickle
		self.custom_model = custom_model
		self.outlier_removal = outlier_removal
		self.outlier_thresh = outlier_thresh
		self.rolling_windows_static()
		self.data_split()
		self.model_selector()

	def rolling_windows_static(self):
		"""Returns rolling windows. If rolling 
		windows exist, just load the file. 
		The `static` suffix means that the 
		windows are pre-specifed and not
		decided dynamically."""
		forecast_df = None
		if self.recalculate:
			print('Calculating rolling windows...')
			forecast_df = lib.create_rolling_windows(self.df, 
				self.WINDOW_SIZE, self.features, self.ohlc)
			self.forecast_df = forecast_df
			return 
		if self.ohlc:
			rolling_data = glob.glob(data_path+'rolling*ohlc*.pkl')
		else:
			rolling_data = glob.glob(data_path+'rolling*last*.pkl')
		for r in rolling_data: 
			if int(r.split('/')[-1].split('_')[1]) == self.WINDOW_SIZE:
				forecast_df = pd.read_pickle(r, compression='zip')
				break
		if forecast_df is None:
			print('Calculating rolling window...')
			forecast_df = lib.create_rolling_windows(self.df, 
				self.WINDOW_SIZE, self.features, self.ohlc, 
				self.save_to_pickle)

		if self.outlier_removal:
			forecast_df = lib.remove_outliers(forecast_df, self.outlier_removal,
									 self.outlier_thresh, self.WINDOW_SIZE)
		self.forecast_df = forecast_df

	def data_split(self):
		"""Split data into training and validation.
		Optional param to create a test set from 
		validation."""
		if self.test_set:
			self.X_train, self.X_valid, self.X_test, self.y_train, \
			self.y_valid, self.y_test = lib.train_test_valid_split(self.forecast_df, 
				self.WINDOW_SIZE, len(self.features), self.split_pct_1, 
				self.split_pct_2, test_set=self.test_set)
		else:
			self.X_train, self.X_valid, self.y_train, self.y_valid = \
			lib.train_test_valid_split(self.forecast_df, 
				self.WINDOW_SIZE, len(self.features), self.split_pct_1, 
				self.split_pct_2, test_set=self.test_set)

	def model_selector(self):
		"""Select and fit the model based on 
		the `regressor` string"""
		regressors = ['randFor', 'gradBoost', 'xgBoost', 'lstm', 'custom']
		if self.regressor not in regressors:
			raise ValueError('Model not defined')
		else:
			if self.regressor == 'randFor':
				self.random_forest()
			elif self.regressor == 'gradBoost':
				self.grad_boost()
			elif self.regressor == 'xgBoost':
				# Don't want to run forecast model for 
				# xgboost because the API call is not
				# compatible with Sklearn. 
				self.xgboost()
				return
			elif self.regressor == 'lstm':
				self.lstm()
			elif self.regressor == 'custom':
				self.model = self.custom_model

			self.forecast_model()

	def random_forest(self):
		"""Initialise a random forest regressor 
		with a centain set of params"""
		if self.params is None:
			self.model = RandomForestRegressor(n_jobs=-1)
		else:
			self.model = RandomForestRegressor(**self.params, n_jobs=-1, 
												random_state=RANDOM_STATE)

	def grad_boost(self):
		"""Initialise a gradient boosted 
		regressor with a centain set of 
		params"""
		if self.params is None:
			self.model = GradientBoostingRegressor(n_jobs=-1)
		else:
			self.model = GradientBoostingRegressor(**self.params, n_jobs=-1, 
													random_state=RANDOM_STATE)

	def xgboost(self): 
		"""An XGBoost Model"""
		xgb_train = xgb.DMatrix(data=self.X_train, label=self.y_train)
		xgb_valid = xgb.DMatrix(data=self.X_valid, label=self.y_valid)
		if self.test_set:
			xgb_test = xgb.DMatrix(data=self.X_test, label=self.y_test)
		evallist = [(xgb_train, 'train'), (xgb_valid, 'eval')]
		print('Fitting model...')
		if self.params is None:
			params = {'objective': 'reg:squarederror', 'eval_metric': 'rmse', 
				'n_estimators': 100}
		else:
			params = self.params
		self.model = xgb.train(params, xgb_train, 25, evallist)
		self.y_pred = self.model.predict(xgb_test)
		self.metrics = lib.calc_reg_metrics(self.y_test, self.y_pred)

	def lstm_model(self):
		"""An LSTM Model"""
		layers = self.lstm_layer
		model = Sequential()
		model.add(LSTM(layers[0], input_shape = self.X_train.shape[1:], return_sequences=True))  # add first layer
		model.add(Dropout(self.dropout))  # add dropout for first layer
		#model.add(RepeatVector(5))
		model.add(LSTM(layers[1],  return_sequences=True))  # add second layer
		model.add(Dropout(self.dropout))  # add dropout for second layer
		model.add(TimeDistributed(Dense(layers[2])))  # add dense layer
		model.add(Activation("relu"))
		model.add(Flatten())
		model.add(Dense(layers[3]))  # add output layer
		model.add(Activation('linear'))  # output layer with linear activation
		model.compile(loss="mae", optimizer="adam", metrics=['mse', 'mae', lib.coeff_determination])
		# model = Sequential()
		# layers = self.lstm_layer
		# model.add(LSTM(layers[0], input_shape=self.X_train.shape[1:], return_sequences=True))  # add first layer
		# model.add(Dropout(self.dropout))  # add dropout for first layer
		# model.add(LSTM(layers[1],  return_sequences=False))  # add second layer
		# model.add(Dropout(self.dropout))  # add dropout for second layer
		# model.add(Dense(layers[2]))  # add output layer
		# model.add(Activation('linear'))  # output layer with linear activation
		# model.compile(loss="mae", optimizer="adam", metrics=['mse', 'mae', lib.coeff_determination])
		return model

	def reshape_for_lstm(self):
		self.X_train = self.X_train.values.reshape(-1, len(self.features), self.WINDOW_SIZE).transpose(0,2,1)
		self.y_train = self.y_train
		self.X_valid = self.X_valid.values.reshape(-1, len(self.features), self.WINDOW_SIZE).transpose(0,2,1)
		self.y_valid = self.y_valid
		self.X_test = self.X_test.values.reshape(-1, len(self.features), self.WINDOW_SIZE).transpose(0,2,1)
		self.y_test = self.y_test

	def lstm(self):
		try:
			self.lstm_layer = self.params['lstm_layer']
			self.batch_size = self.params['batch_size']
			self.dropout = self.params['dropout']
			self.epochs = self.params['epochs']
		except:
			raise ValueError('Did not receive all required \
				parameters for the LSTM.')
		self.reshape_for_lstm()
		self.model = self.lstm_model()

	def forecast_model(self):
		"""Fit the model and caculate performance.
		Only suited for Sk-Learn style-API"""
		if self.grid_search:
			print('Running Grid Search')
			self.model = GridSearchCV(self.model, self.params, n_jobs=-1, verbose=3)
		print('Fitting model...')
		if self.regressor == 'lstm':
			history = self.model.fit(self.X_train, self.y_train, 
						validation_data=(self.X_valid, self.y_valid), 
						batch_size=self.batch_size, epochs=self.epochs)
		else:
			self.model.fit(self.X_train, self.y_train)
		self.y_pred = self.model.predict(self.X_valid)
		self.metrics = lib.calc_reg_metrics(self.y_valid, self.y_pred)

	def feature_importance(self, type_='high'):
		"""Plot feature importance"""
		if self.regressor in ['randFor', 'gradBoost']: 
			if type_=='high':
				feat_importances = pd.Series(self.model.feature_importances_, index=self.X_train.columns).nlargest(20)[::-1]
				feat_importances.plot(kind='barh', figsize=(16,10))
			if type_=='low':
				feat_importances = pd.Series(self.model.feature_importances_, index=self.X_train.columns).nsmallest(20)[::-1]
				feat_importances.plot(kind='barh', figsize=(16,10))
		elif self.regressor == 'xgBoost':
			ax = xgb.plot_importance(self.model)
			fig = ax.figure
			fig.set_size_inches(16,10)
		else:
			return
