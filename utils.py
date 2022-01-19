import pandas as pd
import numpy as np


def calc_distance(x, y):
	""" 
	Calculates the Euclidean distance between two coordinates

	Args:
		x: First coordinate Tuple
		y: Second coordinate Tuple

	Returns:
		Unit distance measure
	"""
	return np.sqrt((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2)


def interpolate(c1, c2, s, t):
	"""
	Interpolates between two coordinates at some given unit speed and time

	Args:
		c1: First coordinate Tuple
		c2: Second coordinate Tuple
		s: Unit speed
		t: Time elapsed

	Returns:
		The new interpolated point or the final point with some epsilon accuracy
	"""
	x1 = c1[0]
	y1 = c1[1]
	x2 = c2[0]
	y2 = c2[1]

	epsilon = .00001

	d = s * t

	diff_y = y2 - y1
	diff_x = x2 - x1

	bearing = np.arctan2(diff_y, diff_x)
	new_point = (x1 + (d * np.cos(bearing)), y1 + (d * np.sin(bearing)))

	original_dist = calc_distance(c1, c2)
	new_dist = calc_distance(c1, new_point)

	if new_dist >= original_dist - epsilon:
		return c2
	else:
		return new_point


def max_value(inputlist, index):
	"""
	Returns the max value of sublist

	Args:
		inputList: Full list containing sublists
		index: The index of the sublists that contains the values that we want the max of
	Returns:
		Max value of the entries of the sublists
	"""
	return max([sublist[index] for sublist in inputlist])


def min_value(inputlist, index):
	"""
	Returns the min value of sublist

	Args:
		inputList: Full list containing sublists
		index: The index of the sublists that contains the values that we want the min of
	Returns:
		Min value of the entries of the sublists
	"""
	return min([sublist[index] for sublist in inputlist])


def sort_sub(sub_list, reverse=False):
	"""
	Sorts a list based on the value of a sublist
	"""
	return(sorted(sub_list, reverse = reverse, key = lambda x: x[1]))


def sum_raw_driver_hours(driver_hours):
	"""
	Calculates the total driver hours from a raw driver hours list (i.e. [[t1, drivers], [t2, drivers], ... [T, drivers]]).
	This is different from when we bin driver hours into equal length bins (removing the time-stamps) and can just sum and 
	multiply by the time per bin.
	"""
	total = 0
	for i in range(len(driver_hours)):
		if i == 0:
			continue
		else:
			delta = (driver_hours[i][0] - driver_hours[i-1][0])
			value = delta*driver_hours[i-1][1]
			total += value

	return total


def make_driver_hours_from_data(df, started='started_at'):
	"""
	Creates a raw driver hours list from historical data. Has the option to choose when to start "counting" driver activity as driver
	hours, for example: started_at vs. arrived_at_pickup.
	"""
	started_times = list((pd.to_datetime(df[started]).dt.minute+(pd.to_datetime(df[started]).dt.hour*60)).values)
	delivered_times = list((pd.to_datetime(df.delivered_at).dt.minute+(pd.to_datetime(df.delivered_at).dt.hour*60)).values)
	s_list = [[1, t] for t in started_times]
	d_list = [[-1, t] for t in delivered_times]
	t_list = s_list + d_list
	t_list = sort_sub(t_list)
	x = [t[1] for t in t_list]
	x.append(1440)
	y = [0 for t in t_list]
	val = 0
	for i in range(len(t_list)):
		val = val+t_list[i][0]
		y[i] = val
	
	y.insert(0, 0)
	y.append(0)
	x.insert(0, 0)

	driver_hours = [[x[i], y[i]] for i in range(len(x))]

	return driver_hours


def get_dh_between(dh, t1, t2):
	"""
	Returns the subset of driver hour sublists that fall in between two time points (t1, t2) from the overall
	driver hour list (dh).
	"""
	temp_dh = [thing for thing in dh if (t1 < thing[0] <= t2)]
	if len(temp_dh) != 0:
		last_index = dh.index(temp_dh[-1])
	else:
		last_index = None
	return temp_dh, last_index


def bin_driver_hours(dh, num_bins, T):
	"""
	Furthers discretizes the driver hour vector into segments of equal length by averaging.

	Args:
		dh: Driver hour vector returned from simulation (list of lists)
		num_bins: The number of bins to divide the time horizon into
		T: Time horizon

	Returns: 
		Binned driver hour vector
	"""
	mins_per_bin = T/num_bins
	bin_list = [0 for _ in range(num_bins)]
	time_list = [(i+1)*mins_per_bin for i in range(num_bins)]
	last_val= dh[0][1]

	for i in range(len(time_list)):
		temp_dh, last_index = get_dh_between(dh, i*mins_per_bin, time_list[i])
		if len(temp_dh) == 0:
			bin_list[i] = last_val
		else:
			weighted_total = 0
			for j in range(len(temp_dh)):
				if j==0:
					w = (temp_dh[j][0]-(i*mins_per_bin))/mins_per_bin
					val = last_val
				else:
					w = (temp_dh[j][0] - temp_dh[j-1][0])/mins_per_bin
					val = temp_dh[j-1][1]
				weighted_total += (w*val)
			w = (time_list[i]-temp_dh[-1][0])/mins_per_bin
			val = temp_dh[-1][1]
			weighted_total += (w*val)
			bin_list[i] = weighted_total
			last_val = dh[last_index][1]

	return bin_list


def bin_driver_hours_max(dh, num_bins, T):
	"""
	Furthers discretizes the driver hour vector into segments of equal length by taking the max in any interval.

	Args:
		dh: Driver hour vector returned from simulation (list of lists)
		num_bins: The number of bins to divide the time horizon into
		T: Time horizon

	Returns:
		Binned driver hour vector
	"""
	mins_per_bin = T/num_bins
	bin_list = [0 for _ in range(num_bins)]
	time_list = [(i+1)*mins_per_bin for i in range(num_bins)]
	last_val= dh[0][1]

	for i in range(len(time_list)):
		temp_dh, last_index = get_dh_between(dh, i*mins_per_bin, time_list[i])
		if len(temp_dh) == 0:
			bin_list[i] = last_val
		else:
			bin_list[i] = max([temp_dh[i][1] for i in range(len(temp_dh))])
			last_val = dh[last_index][1]

	return bin_list

def convert_to_single_minute_bins(original, min_per_bin, T):
	output = [0 for _ in range(T)]
	current_mult = 1
	orig_index = 0
	for i in range(T):
		if i == 0:
			continue
		else:
			if i < min_per_bin*current_mult:
				output[i] = original[orig_index]
			else:
				orig_index += 1
				current_mult += 1
				output[i] = original[orig_index] 

	return output


def get_available_vehicles(vehicle_list):
	""" Returns a list of vehicles that do not have any current tasks """
	available_vehicles = [vehicle for vehicle in vehicle_list if vehicle.current_task is None]

	return available_vehicles


def intersperse(lst, item):
    result = [None, item] * len(lst)
    result[0::2] = lst
    return result


def make_origins(num_origins, r):
    if num_origins == 1:
        return [(0,0)]
    else:
        origin_list = [None for _ in range(num_origins)]
        theta = (360/num_origins)*(np.pi/180)
        for i in range(num_origins):
            angle = (i+1)*theta
            origin_list[i] = (np.cos(angle)*(r/2), np.sin(angle)*(r/2))
    return origin_list


def make_bounds(a, b, mean, std):
    return (a - mean) / std, (b - mean) / std, mean, std

