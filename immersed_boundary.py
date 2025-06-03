##### This script holds functions that assist in the development of an immersed boundary method

import numpy as np
import math

def get_points_on_circle(radius:float, nx:int, ny:int, dx:float, dy:float)-> tuple:
    '''This functions calulates the spatial discretizations of a circle and returns
    the x, y coordinates of each point on its perimeter. The point to point separation will be
    approximatley the same size as the cartesian grid discretization
    The circle is assumed to be centered on nx/2*dx and ny/2*ny
    Returns a list of the (x,y) coordinates for each point on surface'''

    

    center_x = nx*dx/2
    center_y = ny*dy/2

    perimeter = math.pi*2*radius
    avg_grid = (dx+dy)/2

    num_points = round(perimeter / avg_grid)
    angles = np.linspace(0, math.pi*2, num_points)

    x_coords = list(map(lambda x: radius*math.sin(x)+center_x, angles))
    y_coords = list(map(lambda x: radius*math.cos(x)+center_y, angles))

    return (x_coords, y_coords)


def Eu(u_vel, v_vel, nx, ny, dx, dy, x_c, y_c):
    '''This functions takes the points in the staggered grid, and interpolates the velocity
    at the immersed boundary points'''


    pass


