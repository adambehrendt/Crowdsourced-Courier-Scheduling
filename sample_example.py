import utils
import numpy as np
import scipy.stats as stats


def example(inhomogeneous=False):
	T = 720
	S = 60

	num_pickup_bins = 48
	num_periods = 26

	radius = 20
	num_origins = 4
	origin_list = utils.make_origins(num_origins, radius)
	origin_rv = stats.randint(0, num_origins)

	K = 100

	# Make K realizations for a random row (jth) of the dataset
	if not inhomogeneous:
		X = np.genfromtxt('data/data_homogeneous.csv', delimiter=',')
	else:
		X = np.genfromtxt('data/data_inhomogeneous.csv', delimiter=',')

	j = stats.randint(0, len(X))

	pickup_histogram = X[j, 0:num_pickup_bins]
	d_mean = X[j, num_pickup_bins]
	d_std = X[j, num_pickup_bins+1]
	N_mean = X[j, num_pickup_bins+2]
	N_std = X[j, num_pickup_bins+3]
	num_known = X[j, num_pickup_bins+4]
	if not inhomogeneous:
		actual_lambda = X[j, num_pickup_bins+5]
			
	else:
		actual_lambda = sub_X_test[j, num_pickup_bins+5:num_pickup_bins+5+num_periods]

	const_rate = actual_lambda/((T+S)/num_periods)

	a, b, mu, sigma = utils.make_bounds(0, np.inf, d_mean, d_std)
	distance_rv = stats.truncnorm(a, b, loc=mu, scale=sigma)
	a, b, mu, sigma = utils.make_bounds(0, np.inf, N_mean, N_std)
	N_rv = stats.truncnorm(a, b, loc=mu, scale=sigma)

	realization_set = [None for _ in range(evaluation_size)]
	ad_hoc_set = [None for _ in range(evaluation_size)]
	for k in range(K):
		order_list = instance_generator.gen_random_order_list(N_rv, 45, pickup_histogram, S, distance_rv, origin_rv, origin_list, num_known, placement_is_pickup=False)
		order_list.sort(key=lambda x: x.placement_time)
		realization_set[k] = order_list
		if Lambda == 0:
			pass 
		else:
			if not inhomogeneous:
				ad_hoc_set[k] = instance_generator.gen_poisson_arrivals(const_rate, T+S, start=0)
			else:
				ad_hoc_set[k] = instance_generator.gen_inhomogeneous_arrivals(const_rate, T+S, num_periods)

	return realization_set, ad_hoc_set
