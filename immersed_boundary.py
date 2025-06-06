##### This script holds functions that assist in the development of an immersed boundary method

import numpy as np
import math
from numba import jit

def get_points_on_circle(radius:float, center_x:float, center_y:float, ds:float)-> tuple:
    '''This functions calulates the spatial discretizations of a circle and returns
    the x, y coordinates of each point on its perimeter. The point to point separation will be
    approximatley the same size as the cartesian grid discretization
    The circle is assumed to be centered on nx/2*dx and ny/2*ny
    Returns a list of the (x,y) coordinates for each point on surface'''

    # center_x = nx*dx/2
    # center_y = ny*dy/2

    perimeter = math.pi*2*radius

    num_points = round(perimeter / ds)
    #print(f'points on circ {num_points}')
    angles = np.linspace(0, math.pi*2, num_points, endpoint=False)

    x_coords = list(map(lambda x: radius*math.cos(x)+center_x, angles))
    y_coords = list(map(lambda x: radius*math.sin(x)+center_y, angles))

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
            if dist_x <= 1.5*ds and dist_y <= 1.5*ds:
                x_results.append((cart_x[j], la_x[i], dist_x))
                y_results.append((cart_y[j], la_y[i], dist_y))


    return(x_results, y_results)

def calc_q(cart_x:list, cart_y:list, la_x:list, la_y:list, ds:float):
    '''This functions takes an array of points on a cartesian grid and an array points on the langragian grid it
    then computes finds the values in each that are within 1.5*ds distance of one another
    It returns a list of tuples that include the index in q for the cartesian point, x distance and y distance
    (l_index, q_index, x_dist, y_dist)
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
            dist_x = la_x[i]- cart_x[j]
            dist_y = la_y[i]- cart_y[j]
            if abs(dist_x) <= 1.5*ds and abs(dist_y) <= 1.5*ds:
                results.append((i, j, dist_x, dist_y))


    return(results)

def compute_H(circ:list, ds:float) -> list:
    '''Computes the H operator for the immersed boundary method
    circ should contain a list of tuples (index, x_dist, y_dist)
    and H will return the index and force (lagragian  index, cartesian_index force)'''
    H = []
    beta = ds**2
    

    for i in range(len(circ)):

        # print(f'dx {circ[i][1]}')
        # print(f'ds {ds}')
        # print(f'ds/2 {ds/2}')
        # print(f'ds/ds = {circ[i][1]/ds}')

        # Compute d3 related to distances
        
        d3_x = d3(circ[i][2], ds)
        d3_y = d3(circ[i][3], ds)
    

        H.append((circ[i][0], circ[i][1], beta*d3_x*d3_y)) 
        
    return H

@jit()
def HF(F, circ, ds, length):
    '''Computes HF'''

    HF = np.zeros(length)

    beta = ds**2
    

    for point in circ:
        
        HF[point[1]] += beta*F[point[0]]*d3(point[2], ds)*d3(point[3], ds)
        
        
    return HF


@jit()
def d3(dist:float, ds:float):
    '''Computes d3'''
    if abs(dist) <= ds*.5:
        d_3 = 1/3*(1+math.sqrt(-3 * (abs(dist)/ds)**2 + 1))
    else:
        d_3 = 1/6*(5 - 3*abs(dist)/ds - math.sqrt(-3 * (1- abs(dist)/ds)**2 + 1) )

    #print(ds)
    d_3 = d_3/ds
    return d_3


@jit()
def Eu(u_vels, circ, ds):
    '''Interpolates the velocities from the cartesian grid to the Lagragian'''

    #print(circ[-1][0])
    q_b = np.zeros(circ[-1][0]+1)
    alpha = ds**2
    # alpha = 1
    

    for point in circ:
        #print(point[1])
        q_b[point[0]] += alpha*u_vels[point[1]]*d3(point[2],ds)*d3(point[3],ds)

    return q_b



# def Eu(q, E_u, E_v):
#     '''Interpolates the velocities from the cartesian boundary to the langragian boundary'''
#     q_b = np.zeros(2*E_u[-1][0])
#     # print(f'q_b size = {2*E_u[-1][0]}')
#     # print(f'q_b len {len(q_b)}')
#     # U vels
#     for point in E_u:
#         #print(point[0])
#         q_b[point[0]] += (q[point[1]]*point[1])

#     # V vels
#     for point in E_v:
#         #print(point[0])
#         q_b[point[0]] += (q[point[1]]*point[1])

#     return q_b

# def compute_E(circ:list, ds:float)->list:
#     '''Compute the interpolated velocity from the cartesian grid onto the lagragian points
#     E will return the index and force (lagragian  index, cartesian_index force)'''
#     E = []

#     alpha = ds
#     for i in range(len(circ)):
#         d3_x = d3(circ[i][2], ds)
#         d3_y = d3(circ[i][2], ds)
    
#         E.append((circ[i][0], circ[i][1], alpha*d3_x*d3_y))

#     return E