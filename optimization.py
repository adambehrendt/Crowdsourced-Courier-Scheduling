import gurobipy as gp
from itertools import zip_longest
import numpy as np
import simulation
import time
import random
from copy import copy, deepcopy

def pair_iterator(event):
    event = [0] + event

    return enumerate(zip_longest(event, event[1:], fillvalue=0), -1)

def one_ranges(event):
    start = [i + 1 for i, pair in pair_iterator(event) if pair == (0, 1)]
    end = [i for i, pair in pair_iterator(event) if pair == (1, 0)]

    return zip(start, end)


def make_shift_model(P, L_min, L_max):
	"""
	Instantiates the IP that solves for the shifts.
	"""
	model = gp.Model(name='scheduling')
	model.Params.OutputFlag = 0

	num_shifts = sum([len(P)-(val-1) for val in range(L_min, L_max+1, 1)])
	l = [[0 for _ in P] for i in range(num_shifts)]
	count = 0
	for val in range(L_min, L_max+1, 1):
		for start in range(len(P)-(val-1)):
			l[count][start:start+val] = [1 for _ in range(val)]
			count += 1
	S = range(len(l))
	u = {s: model.addVar(name='u_{s}'.format(s=s), obj = 0, vtype=gp.GRB.INTEGER, lb=0) for s in S}

	return model, l, u

def solve_shift_model(model, u, z, l, P):
	"""
	Finds a minimum cost set of shifts that covers the required number of scheduled couriers.
	"""
	S = range(len(l))
	model.setObjective(gp.quicksum(gp.quicksum(l[s][p]*u[s] for s in S) for p in P) + gp.quicksum(u[s] for s in S), gp.GRB.MINIMIZE)
	for p in P:
		model.addConstr(gp.quicksum(l[s][p]*u[s] for s in S) >= z[p])

	model.optimize()

	if model.status == 2:
		v = model.getVars()
		results = [(0,0) for i in range(len(v))]
		for s in range(len(v)):
			results[s] = (v[s].varName, v[s].X)

		u = [v[s].X for s in S]
		indices = [(i, int(x)) for i, x in enumerate(u) if x > 0]
		optimal_shifts = [None for _ in range(sum(indices[i][1] for i in range(len(indices))))]
		count = 0
		for i in indices:
			for j in range(i[1]):
				optimal_shifts[count] = l[i[0]]
				count += 1

		model.reset()

		return optimal_shifts

	else:
		print('Not optimal...')
		quit()	


def max_sum_n(lst, n):
	best_idx = 0
	best_sum = 0
	for i in range(len(lst)):
		temp = sum(lst[i:i+n+1])
		if temp > best_sum:
			best_sum = temp
			best_idx = i

	return best_idx, best_sum


def reset_realization_set(realization_set):
	for order_set in realization_set:
		for order in order_set:
			order.attempted_vehicle_inserts = []
			order.assigned = False
			order.actual_pickup_time = None
			order.actual_delivery_time = None


def solve_SAA(T, S_max, V, service_level, penalty_cost, fixed_wage, origin_list, realization_set, num_periods, L_min, L_max, num_non_improve, ad_hoc_wage, ad_hoc_set, start_z = None):

	new_T = T + S_max
	time_per_period = new_T/num_periods
	P = range(num_periods)
	K = len(realization_set)

	# If no initial z is given, start from scratch
	if start_z is None:
		z = [0 for _ in range(num_periods)]
		best_z = None
		old_cost = 500000000000
		best_expired = 500000000000
		best_ad_hoc = 500000000000
	# Id an initial z is given, first evaluate it to start
	else:
		z = start_z
		best_z = start_z
		model, l, u = make_shift_model(P, L_min, L_max)
		optimal_shifts = solve_shift_model(model, u, z, l, P)

		shift_list = [None for _ in range(len(optimal_shifts))]
		for i in range(len(optimal_shifts)):
			shift_indices = [thing for thing in one_ranges(optimal_shifts[i])][0]
			shift_tuple = (shift_indices[0]*time_per_period, (shift_indices[1]+1)*time_per_period)
			shift_list[i] = shift_tuple

		fixed_cost = fixed_wage*sum([sum(shift) for shift in optimal_shifts])
		expired_cost = np.zeros(K)
		ad_hoc_cost = np.zeros(K)
		period_score = np.zeros((K, num_periods))
		for k in range(K):
			expired_cost[k], ad_hoc_cost[k], period_score[k], expired_perc = simulation.run_sim(new_T, V, service_level, penalty_cost, shift_list, origin_list, realization_set[k], num_periods, ad_hoc_wage, ad_hoc_set[k])

		reset_realization_set(realization_set)

		total_period_score = np.sum(period_score, axis=0)
		avg_cost = np.mean(expired_cost)
		total_cost = fixed_cost + avg_cost

		old_cost = total_cost
		best_shift_list = shift_list 
		best_cost = old_cost 
		best_fixed = fixed_cost
		best_expired = avg_cost
		best_ad_hoc = np.mean(ad_hoc_cost)


	# Initialize the agorithm
	non_improve_count = 0
	max_non_improve = num_non_improve
	iteration = 0
	model, l, u = make_shift_model(P, L_min, L_max)

	while True:
		iteration += 1
		t0 = time.time()

		# Make shifts for the incumbent z and get fixed scheduled cost
		model, l, u = make_shift_model(P, L_min, L_max)
		optimal_shifts = solve_shift_model(model, u, z, l, P)
		shift_list = [None for _ in range(len(optimal_shifts))]
		for i in range(len(optimal_shifts)):
			shift_indices = [thing for thing in one_ranges(optimal_shifts[i])][0]
			shift_tuple = (shift_indices[0]*time_per_period, (shift_indices[1]+1)*time_per_period)
			shift_list[i] = shift_tuple
		fixed_cost = fixed_wage*sum([sum(shift) for shift in optimal_shifts])

		# Evaluate the uncertain cost of our shift soluton by simulation, over the K realizations
		expired_cost = np.zeros(K)
		ad_hoc_cost = np.zeros(K)
		period_score = np.zeros((K, num_periods))
		pickup_score = np.zeros((K, num_periods))
		for k in range(K):
			expired_cost[k], ad_hoc_cost[k], period_score[k], expired_perc = simulation.run_sim(new_T, V, service_level, penalty_cost, shift_list, origin_list, realization_set[k], num_periods, ad_hoc_wage, ad_hoc_set[k])
		reset_realization_set(realization_set)

		total_period_score = np.sum(period_score, axis=0)
		avg_cost = np.mean(expired_cost)
		total_cost = fixed_cost + avg_cost

		# If the solution hasn't improved, update the counter of non improving steps
		if total_cost >= old_cost:
			non_improve_count += 1
		# Otherwise, update our "best" z
		else:
			best_z = deepcopy(z)
			best_shift_list = deepcopy(shift_list)
			best_cost = deepcopy(total_cost)
			best_fixed = deepcopy(fixed_cost)
			best_expired = deepcopy(avg_cost)
			best_ad_hoc = np.mean(ad_hoc_cost)
			old_cost = deepcopy(total_cost)
			non_improve_count = 0

		# Now lets update z by using information from our "pseudo gradient"
		if non_improve_count == max_non_improve:
			# If we have reached the allotted number of non-improving steps, take our best solution and terminate
			break
		else:
			# Here width represents how many consecutive periods we will update and delta represents how many requried couriers we will add to the periods
			# When width=1 and delta=1, we move by one courier in one period at a time.
			# Here we provide an example where depending on the number of expired orders (the array we refer to as the total_period_score), the width and delta changes
			# As iterations continue, we eventually converge to width=1 and delta=1
			width = max(1, int(np.ceil((sum(total_period_score)/num_periods)/(.75*K))))
			idx, val = max_sum_n(total_period_score, width)
			delta = max(1, int(np.floor(val/(50*K))))
			for i in range(idx, idx+width):
				if i >= len(z):
					pass
				else:
					z[i] += delta

		t1 = time.time()

	return best_z, best_shift_list, best_cost, best_fixed, best_expired, best_ad_hoc





