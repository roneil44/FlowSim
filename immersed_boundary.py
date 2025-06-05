##### This script holds functions that assist in the development of an immersed boundary method

import numpy as np
import math

def get_points_on_circle(radius:float, nx:int, ny:int, dx:float, dy:float, ds:float)-> tuple:
    '''This functions calulates the spatial discretizations of a circle and returns
    the x, y coordinates of each point on its perimeter. The point to point separation will be
    approximatley the same size as the cartesian grid discretization
    The circle is assumed to be centered on nx/2*dx and ny/2*ny
    Returns a list of the (x,y) coordinates for each point on surface'''

    

    center_x = nx*dx/2
    center_y = ny*dy/2

    perimeter = math.pi*2*radius

    num_points = round(perimeter / ds)
    angles = np.linspace(0, math.pi*2, num_points)

    x_coords = list(map(lambda x: radius*math.sin(x)+center_x, angles))
    y_coords = list(map(lambda x: radius*math.cos(x)+center_y, angles))

    return (x_coords, y_coords)


def calc_influence(cart_x:list, cart_y:list, la_x:list, la_y:list, ds:float):
    '''This functions takes an array of points on a cartesian grid and an array points on the langragian grid it
    then computes finds the values in each that are within 2.5*ds distance of one another
    It returns a list of tuples that include the locations in space for the cartesian point, x distance and y distance
    (x_catesian_location, x_lagragian_location, x_dist), (y_catesian_location, y_lagragian_location, y_dist)
    '''

    # Iterate through each lagragian point and identify which locations on thhe cartesian grid are nearby

    x_results = []
    # for lag_point in la_x:
    #     for cart_point in cart_x:
    #         dist = abs(cart_point-lag_point)
    #         if  dist <= 2.5*ds:
    #             x_results.append((cart_point, lag_point, dist))

    y_results = []
    # for lag_point in la_y:
    #     for cart_point in cart_y:
    #         dist = abs(cart_point-lag_point)
    #         if  dist <= 2.5*ds:
    #             y_results.append((cart_point, lag_point, dist))
    
    for i in range(len(la_x)):
        for j in range(len(cart_x)):
            dist_x = abs(la_x[i]- cart_x[j])
            dist_y = abs(la_y[i]- cart_y[j])
            if dist_x <= 2.5*ds and dist_y <= 2.5*ds:
                x_results.append((cart_x[j], la_x[i], dist_x))
                y_results.append((cart_y[j], la_y[i], dist_y))


    return(x_results, y_results)

def calc_q(cart_x:list, cart_y:list, la_x:list, la_y:list, ds:float):
    '''This functions takes an array of points on a cartesian grid and an array points on the langragian grid it
    then computes finds the values in each that are within 2.5*ds distance of one another
    It returns a list of tuples that include the index in q for the cartesian point, x distance and y distance
    (q_index, x_dist, y_dist)
    '''

    # Iterate through each lagragian point and identify which locations on thhe cartesian grid are nearby

    results = []
    # for lag_point in la_x:
    #     for cart_point in cart_x:
    #         dist = abs(cart_point-lag_point)
    #         if  dist <= 2.5*ds:
    #             x_results.append((cart_point, lag_point, dist))
    # for lag_point in la_y:
    #     for cart_point in cart_y:
    #         dist = abs(cart_point-lag_point)
    #         if  dist <= 2.5*ds:
    #             y_results.append((cart_point, lag_point, dist))
    
    for i in range(len(la_x)):
        for j in range(len(cart_x)):
            dist_x = abs(la_x[i]- cart_x[j])
            dist_y = abs(la_y[i]- cart_y[j])
            if dist_x <= 2.5*ds and dist_y <= 2.5*ds:
                results.append((j, dist_x, dist_y))


    return(results)


