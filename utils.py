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


def make_origins(num_origins, r):
	"""
    Function that evenly spread origins (depots, stores, etc.) In a circle with a given radius.
    """

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
	"""
	Makes the bounds to be used for a truncated normal distribution.
	"""

    return (a - mean) / std, (b - mean) / std, mean, std

