import utils
import numpy as np
import scipy.stats as stats
import instance_generator
import optimization
import prediction
import pickle


def example(inhomogeneous=False):

	### INITIALIZE ###
	T = 720
	S = 60
	V = 1
	num_pickup_bins = 48
	num_periods = 26
	radius = 20
	num_origins = 4
	origin_list = utils.make_origins(num_origins, radius)
	origin_rv = stats.randint(0, num_origins)

	if not inhomogeneous:
		X = np.genfromtxt('data/data_homogenous.csv', delimiter=',')
	else:
		X = np.genfromtxt('data/data_inhomogenous.csv', delimiter=',')

	### GENERATING INSTANCES ###

	# Each row of the data set X.csv is composed of distributional information.
	# Lets use a single row (representing the distribution of an operational period) to generate some
	# realizations of order sets and ad-hoc courier arrival sequences.

	K = 50
	j = np.random.randint(0, len(X))

	# Extract distributional info
	pickup_histogram = X[j, 0:num_pickup_bins]
	d_mean = X[j, num_pickup_bins]
	d_std = X[j, num_pickup_bins+1]
	N_mean = X[j, num_pickup_bins+2]
	N_std = X[j, num_pickup_bins+3]
	num_known = X[j, num_pickup_bins+4]
	if not inhomogeneous:
		actual_lambda = X[j, num_pickup_bins+5]
			
	else:
		actual_lambda = X[j, num_pickup_bins+5:num_pickup_bins+5+num_periods]

	const_rate = actual_lambda/((T+S)/num_periods)

	# Construct truncated normal RV's for the O-D distance and number of dynamic orders
	a, b, mu, sigma = utils.make_bounds(0, np.inf, d_mean, d_std)
	distance_rv = stats.truncnorm(a, b, loc=mu, scale=sigma)
	a, b, mu, sigma = utils.make_bounds(0, np.inf, N_mean, N_std)
	N_rv = stats.truncnorm(a, b, loc=mu, scale=sigma)

	# Using helper functions from "instance_generator.py", generate a set of order sets and a set of arrival sequences
	realization_set = [None for _ in range(K)]
	ad_hoc_set = [None for _ in range(K)]
	for k in range(K):
		order_list = instance_generator.gen_random_order_list(N_rv, 45, pickup_histogram, S, distance_rv, origin_rv, origin_list, num_known)
		order_list.sort(key=lambda x: x.placement_time)
		realization_set[k] = order_list
		if not inhomogeneous:
			ad_hoc_set[k] = instance_generator.gen_poisson_arrivals(const_rate, T+S, start=0)
		else:
			ad_hoc_set[k] = instance_generator.gen_inhomogeneous_arrivals(const_rate, T+S, num_periods)

	### RUNNING SAA-SO METHOD ###

	# Here, we use the K realizations of demand/ad-hoc arrivals to construct a solution z (the number of required scheduled couriers)
	# The details of the method are in "optimization.py"
	service_level = 1.0
	fixed_wage = 10
	ad_hoc_wage = 20
	penalty_cost = 200
	L_min = 4
	L_max = 12
	num_non_improve = 10

	#z, shift_list, total_cost, fixed_cost, expired_cost, ad_hoc_cost = optimization.solve_SAA(T, S, V, service_level, penalty_cost, fixed_wage, origin_list, realization_set, num_periods, L_min, L_max, num_non_improve, ad_hoc_wage, ad_hoc_set)

	#print(z)

	### MACHINE LEARNING ###

	# By generating a set of solutions (z) for each row of X.csv we create a data set for our target variable.
	# For illustrative purposes, lets take a very small (100 samples) training set of pre-made solutions
	# to train our model on.

	history_length = 25
	es = prediction.MyThresholdCallback(threshold=.05)

	if not inhomogeneous:
		X_train = pickle.load(open('data/X_train_small.p', 'rb'))
		y_train = pickle.load(open('data/K50_y_train_small.p', 'rb'))

		ANN_model = prediction.make_one_shot_nn(X_train, None, y_train, None, num_pickup_bins+5+1, num_periods, 200, [100, 100], es=es)
		ANN_predictor = prediction.DiscretePredictor(num_pickup_bins+5+1, num_periods, ANN_model, 'one_shot')

		X_roll, y_roll = prediction.make_rolling_data(X_train, y_train, num_pickup_bins+5+1, history_length)
		RANN_model = prediction.make_rolling_nn(X_roll, None, y_roll, None, num_pickup_bins+5+1+history_length+1+1, 20, [100, 100], es=es)
		RANN_predictor = prediction.DiscretePredictor(num_pickup_bins+5+1, num_periods, RANN_model, 'rolling')

	else:
		X_train = pickle.load(open('data/X_train_small_inhomogenous.p', 'rb'))
		y_train = pickle.load(open('data/K50_y_train_small_inhomogenous.p', 'rb'))

		ANN_model = prediction.make_one_shot_nn(X_train, None, y_train, None, num_pickup_bins+5+num_periods, num_periods, 200, [100, 100], es=es)
		ANN_predictor = prediction.DiscretePredictor(num_pickup_bins+5+num_periods, num_periods, ANN_model, 'one_shot')

		X_roll, y_roll = prediction.make_rolling_data(X_train, y_train, num_pickup_bins+5+num_periods, history_length)
		RANN_model = prediction.make_rolling_nn(X_roll, None, y_roll, None, num_pickup_bins+5+num_periods+history_length+1+1, 20, [100, 100], es=es)
		RANN_predictor = prediction.DiscretePredictor(num_pickup_bins+5+num_periods, num_periods, RANN_model, 'rolling')

	ANN_pred_test = ANN_predictor.predict(X, None)
	ANN_pred_ceil = np.ceil(ANN_pred_test)
	ANN_pred_round = np.around(ANN_pred_test)

	print('here')
	RANN_pred_test = np.zeros((len(X), num_periods))
	for i in range(len(X)):
		RANN_pred_test[i] = RANN_predictor.predict(X[i], history_length)
	RANN_pred_ceil = np.ceil(RANN_pred_test)
	RANN_pred_round = np.around(RANN_pred_test)

example(inhomogeneous=False)
