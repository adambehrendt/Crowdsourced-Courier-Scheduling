import numpy as np
import scipy.stats as stats
import random


class Order():
    """
    The class for how we store the properties of an individual order.
    """
    def __init__(self, placement_time, pickup_time, deadline, origin_idx, pickup_loc, delivery_loc, index):
        self.placement_time = placement_time
        self.pickup_time = pickup_time
        self.deadline = deadline
        self.origin_idx = origin_idx
        self.pickup_loc = pickup_loc
        self.delivery_loc = delivery_loc
        self.actual_pickup_time = None
        self.actual_delivery_time = None
        self.index = index
        self.assigned = False
        self.attempted_vehicle_inserts = []


def gen_poisson_arrivals(rate, T, start=0):
    """
    Generate a poisson arrival sequence.
    """
    k = 0
    t = start
    arrivals = []
    while True:
        r = np.random.uniform()
        t = t - (np.log(r)/rate)
        if t > T+start:
            break
        else:
            k = k+1
            arrivals.append(t)

    return arrivals


def gen_inhomogeneous_arrivals(rate, T, num_periods):
    """
    Generate an inhomogenous poisson arrival sequence.
    """
    period_length = T/num_periods
    arrivals = []
    for p in range(num_periods):
        temp = gen_poisson_arrivals(rate[p], period_length, period_length*p)
        arrivals.extend(temp)

    return arrivals


def gen_random_order_list(N_rv, placement_rv, pickup_histogram, window_rv, distance_rv, origin_rv, origin_list, num_known):
    """
    Given distributional information, generate a list of orders.
    """
    num_known = int(num_known)

    if type(N_rv) != int:
        N = int(N_rv.rvs(size=1)) + num_known
    else:
        N = N_rv + num_known

    period_times = [15*i for i in range(len(pickup_histogram))]
    pickup_rv = stats.rv_discrete(name='pickup', values=(period_times, pickup_histogram))
    temp_pickup_times = pickup_rv.rvs(size=N)
    add_term = np.random.randint(0, 15, N)
    pickup_times = temp_pickup_times + add_term

    if type(placement_rv) != int:
        placement_times = placement_rv.rvs(size=N)
    else:
        placement_times = [max(0, pickup_times[i] - placement_rv) for i in range(N)]

    if type(window_rv) != int:
        window_lengths = window_rv.rvs(size=N)
    else: 
        window_lengths = [window_rv for _ in range(N)]

    distance_vals = distance_rv.rvs(size=N)
    origin_vals = origin_rv.rvs(size=N)
    origin_locs = [origin_list[index] for index in origin_vals]
    delivery_locs = [None for _ in range(N)]
    for i in range(N):
        rand_angle = random.random()*np.pi*2
        delivery_locs[i] = (origin_locs[i][0]+np.cos(rand_angle)*distance_vals[i], origin_locs[i][1]+np.sin(rand_angle)*distance_vals[i])

    order_list = [None for _ in range(N)]
    for i in range(N):
        order_list[i] = Order(placement_times[i], pickup_times[i], pickup_times[i]+window_lengths[i], origin_vals[i], origin_locs[i], delivery_locs[i], i)

    known_idx = random.sample(range(N), num_known)
    for idx in known_idx:
        order_list[idx].placement_time = 0

    return order_list
