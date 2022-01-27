import numpy as np
import utils
from copy import copy
from itertools import groupby
import time as clock
import random


class Vehicle():
    """
    Stores information about a particular vehicle
    """

    def __init__(self, index, origin_list, start_loc, speed, start_time, end_time):
        self.index = index
        self.start_time = start_time
        self.end_time = end_time
        self.origin_list = origin_list
        self.current_loc = (0, 0)
        self.route = []
        self.route_index = 0
        self.speed = speed
        self.wait_time = 0

    def take_step(self, current_time, time_jump, order_list):
        """
        Determines and executes vehicle action based on attributes. Gets next location and updates vehicle attributes.

        Args: 
            current_time: Current time in the simulation
            time_jump: Length of time before the next simulation event (the time block we are moving through currently)
            order_list: The global task list of the simulation. Allows us to update this as we complete/pickup tasks

        Returns:
            Nothing. Simply updates vehicle attributes and global task list appropriately.
        """

        # Do the route. if there is no route or you've completed it then we should go to the center of the service region
        if self.route_index == len(self.route):
            self.current_loc = utils.interpolate(self.current_loc, (0, 0), self.speed, time_jump)
        else:
            next_waypoint = self.route[self.route_index]
            previous_wps = self.route[0:self.route_index]
            # Checking if we have to wait at the location
            if self.current_loc in [prev_wp[1] for prev_wp in previous_wps]:
                latest_pickup_time = max([prev_wp[2] for prev_wp in previous_wps])
                self.wait_time = max(0, latest_pickup_time - current_time)
            # If dont need to wait then make the delivery and start next order
            if self.wait_time == 0:
                self.current_loc = utils.interpolate(self.current_loc, next_waypoint[1], self.speed, time_jump)
                if self.current_loc == next_waypoint[1]:
                    # If the following waypoints are at the same location do those actions
                    num_wp = 1
                    for wp in self.route[self.route_index+1:]:
                        if wp[1] == next_waypoint[1]:
                            num_wp += 1
                        else:
                            break

                    grouped_waypoints = self.route[self.route_index:self.route_index+num_wp]

                    for wp in grouped_waypoints:
                        if wp[0]:
                            order_list[wp[3]].actual_pickup_time = current_time + time_jump
                        else:
                            order_list[wp[3]].actual_delivery_time = current_time + time_jump
                    self.route_index += num_wp
                else:
                    pass
            # If you have to wait then do nothing.
            else:
                pass



def tt(x, y, V):
    # "time to" function, for location x to y
    return utils.calc_distance(x, y)/V


def cheapest_insertion(R, current_time, current_loc, V, order, vehicle_id, shift_end):
    """
    Given the route of a vehicle and its current location, find the cheapest place to insert the pickup and delivery of an order.
    """
    # order = [(1, (x, y), e, id), (0, (x, y), l, id)]
    # 1 for pickup 0 for delivery, (x,y) location, e is placement time, l is deadline, id is id of order
    if len(R) == 0:
        final_result = [[(0, 1), tt(current_loc, order[0][1], V)+tt(current_loc, order[1][1], V), vehicle_id]]
    else:
        R.insert(0, (1, current_loc, 0, None))
        potential_pickup_insertions = range(1, len(R)+1, 1)
        feasible_pickup_insertions = insertion_helper(potential_pickup_insertions, R, order[0], V, shift_end, current_time)
        final_result = []
        for item in feasible_pickup_insertions:
            i = item[0]
            R_temp_1 = copy(R)
            R_temp_1.insert(i, order[0])
            potential_delivery_insertions = range(i+1, len(R_temp_1)+1, 1)
            feasible_delivery_insertions = insertion_helper(potential_delivery_insertions, R_temp_1, order[1], V, shift_end, current_time)
            partial_result = [[(i, entry[0]), item[1]+entry[1], vehicle_id] for entry in feasible_delivery_insertions]
            final_result.extend(partial_result)
        R.pop(0)
    
    return final_result


def insertion_helper(insertion_points, R, point, V, shift_end, current_time):
    """
    Helper function for the cheapest insertion
    """

    n = len(R)
    feasible_insertions = []
    for i in insertion_points:
        T_temp = [0 for _ in range(n+1)]
        T_temp[0] = current_time
        R_temp_2 = copy(R)
        R_temp_2.insert(i, point)
        infeasible = False
        for j in range(1, n+1, 1):
            tt_temp = tt(R_temp_2[j-1][1], R_temp_2[j][1], V)
            T_temp[j] = T_temp[j-1] + tt_temp + max(0, R_temp_2[j][0]*R_temp_2[j][2]-tt_temp-T_temp[j-1])
            if not R_temp_2[j][0]:
                if R_temp_2[j][2] < T_temp[j]:
                    infeasible = True
                    break
        if infeasible:
            # Saying we violate a deadline
            continue
        elif T_temp[n] >= shift_end:
            # Saying we violate the shift constraint
            continue
        else:
            if i == n:
                delta_cost = tt(R[i-1][1], point[1], V)
            else:
                delta_cost = (tt(R[i-1][1], point[1], V) + tt(point[1], R[i][1], V)) - tt(R[i-1][1], R[i][1], V)
            feasible_insertions.append([i, delta_cost])
    return feasible_insertions


def assign_to_vehicles(vehicle_list, order, current_time):
    """
    Assigns an order to the vehicle in the vehicle_list that can accomplish it the cheapest.
    """

    inserts = []
    order_info = [(1, order.pickup_loc, order.pickup_time, order.index), (0, order.delivery_loc, order.deadline, order.index)]
    for v in vehicle_list:
        if v.index in order.attempted_vehicle_inserts:
            pass
        else:
            if v.start_time == current_time:
                inserts.extend(cheapest_insertion(v.route, current_time, order_info[0][1], v.speed, order_info, vehicle_list.index(v), v.end_time))
            else:
                inserts.extend(cheapest_insertion(v.route, current_time, v.current_loc, v.speed, order_info, vehicle_list.index(v), v.end_time))
    if len(inserts) != 0:
        idxs, cost, v_idx = min(inserts, key=lambda x: x[1])
        if vehicle_list[v_idx].start_time == current_time:
            vehicle_list[v_idx].current_loc = order_info[0][1]
            vehicle_list[v_idx].route.insert(0, order_info[1])
        else:
            vehicle_list[v_idx].route.insert(idxs[0]-1, order_info[0])
            vehicle_list[v_idx].route.insert(idxs[1]-1, order_info[1])
        return True
    else:
        return False
    

def run_sim(T, V, service_level, penalty_cost, shift_list, origin_list, order_list, num_periods, ad_hoc_wage, ad_hoc_list):
    """
    Agent based, event driven simulation which returns the cost of expired orders, ad-hoc couriers and provides
    the number of orders that expired in each time period
    """

    period_length = T/num_periods

    N = len(order_list)
    # Order added when they expire
    expired_order_list = []
    ad_hoc_order_list = []
    dynamic_order_list = [order for order in order_list if order.placement_time > 0]
    dynamic_order_list.sort(key=lambda x: x.placement_time)
    known_unassigned_order_list = [order for order in order_list if order.placement_time == 0]
    # Vehicle Stuff
    vehicle_list = [Vehicle(i, origin_list, 0, V, shift_list[i][0], shift_list[i][1]) for i in range(len(shift_list))]
    active_vehicle_list = []

    time = 0.0

    while True:

        ### EXPIRED ORDERS CHECK ###

        new_epired_orders = [order_raw for order_raw in known_unassigned_order_list if order_raw.deadline <= time]
        expired_order_list.extend(new_epired_orders)
        for expired_order in new_epired_orders:
            known_unassigned_order_list.remove(expired_order)

        ### UPDATE ACTIVE VEHICLE LIST ###

        active_vehicle_list = [v for v in vehicle_list if v.start_time <= time < v.end_time]

        ### TERMINATION CHECK ###

        if time == T:
            expired_cost = penalty_cost*max((len(expired_order_list) - np.floor((1-service_level)*N)), 0.0)
            ad_hoc_cost = ad_hoc_wage*len(ad_hoc_order_list)
            period_score = [0 for _ in range(num_periods)]
            pickup_score = [0 for _ in range(num_periods)]
            for order in expired_order_list:
                pickup_period = int(np.floor(order.pickup_time/period_length))
                pickup_score[pickup_period] += 1
                deadline_period = int(np.floor(order.deadline/period_length))
                for i in range(pickup_period, deadline_period+1, 1):
                    period_score[i] += 1

            return expired_cost, ad_hoc_cost, period_score, 100*(len(expired_order_list)/N)


        ### ASSIGNMENT BLOCK ###

        # First try and assign dynamic orders

        dynamic_realized_orders = [order_raw for order_raw in dynamic_order_list if order_raw.placement_time == time]
        dynamic_unassigned = []
        for order_raw in dynamic_realized_orders:
            dynamic_order_list.remove(order_raw)
            assigned = assign_to_vehicles(active_vehicle_list, order_raw, time)
            if not assigned:
                dynamic_unassigned.append(order_raw)
                order_raw.attempted_vehicle_inserts.extend([v.index for v in active_vehicle_list])
            else:
                order_raw.assigned = True

        # Next try and assign all old orders

        old_realized_orders = [order_raw for order_raw in known_unassigned_order_list]
        old_realized_orders.sort(key=lambda x: abs(time-x.pickup_time))
        for order_raw in old_realized_orders:
            assigned = assign_to_vehicles(active_vehicle_list, order_raw, time)
            if not assigned:
                order_raw.attempted_vehicle_inserts.extend([v.index for v in active_vehicle_list])
            else:
                known_unassigned_order_list.remove(order_raw)
                order_raw.assigned = True

        # Add dynamic unassigned orders to the list of known unassigned orders

        known_unassigned_order_list.extend(dynamic_unassigned)

        ### AD-HOC BLOCK ###

        if ad_hoc_list is None:
            pass
        else:
            num_arrivals = len([val for val in ad_hoc_list if val == time])
            if num_arrivals != 0:
                for arrival in range(num_arrivals):
                    rand_angle = random.random()*np.pi*2
                    ad_hoc_distance = np.random.uniform(0, 20)
                    ad_hoc_loc = (0+np.cos(rand_angle)*ad_hoc_distance, 0+np.sin(rand_angle)*ad_hoc_distance)
                    num_origins = len(origin_list)
                    dist_list = [utils.calc_distance(ad_hoc_loc, origin_list[i]) for i in range(num_origins)]
                    idx = dist_list.index(min(dist_list))
                    feasible_orders = [order_raw for order_raw in known_unassigned_order_list if order_raw.origin_idx == idx]
                    feasible_orders = [order_raw for order_raw in feasible_orders if tt(ad_hoc_loc, order_raw.pickup_loc, V)+tt(order_raw.pickup_loc, order_raw.delivery_loc, V) <= order_raw.deadline - time]

                    idxs = random.sample(range(len(feasible_orders)), min(len(feasible_orders), 1))
                    for idx in sorted(idxs, reverse=True):
                        feasible_orders[idx].assigned = True
                        ad_hoc_order_list.append(feasible_orders[idx])
                        known_unassigned_order_list.remove(feasible_orders[idx])

            else:
                pass

        ### EVENT BASED TIME JUMP BLOCK ###

        # Time until end of horizon
        time_jump = T-time

        # Time until next task becomes available
        if len(dynamic_order_list) != 0:
            next_order_time = dynamic_order_list[0].placement_time - time
            if 0 < next_order_time < time_jump:
                time_jump = next_order_time

        # Time until next ad hoc arrival
        if ad_hoc_list is not None:
            upcoming_ad_hoc_list = [val for val in ad_hoc_list if val > time]
            if len(upcoming_ad_hoc_list) != 0:
                next_driver_time = upcoming_ad_hoc_list[0] - time
                if 0 < next_driver_time < time_jump:
                    time_jump = next_driver_time

        # Time until vehicles reach next waypoint
        for v in active_vehicle_list:
            if v.route_index < len(v.route):
                next_waypoint_time = tt(v.current_loc, v.route[v.route_index][1], v.speed)
                if 0 < next_waypoint_time < time_jump:
                    time_jump = next_waypoint_time

        # Time until next shift start or end
        for v in vehicle_list:
            time_till_start = v.start_time - time
            if 0 < time_till_start < time_jump:
                time_jump = time_till_start

            time_till_end = v.end_time - time
            if 0 < time_till_end < time_jump:
                time_jump = time_till_end

        for v in active_vehicle_list:
            time_to_wait = v.wait_time
            if 0 < time_to_wait < time_jump:
                time_jump = time_to_wait
        

        ### VEHICLE ACTION ###

        for vehicle in vehicle_list:
            vehicle.take_step(time, time_jump, order_list)

        ### TIME JUMP ###

        time += time_jump


