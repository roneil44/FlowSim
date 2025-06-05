######
# Main file for incompressible flow solver
# #######



## Imports
from vector import *

import numpy as np
from matplotlib import pyplot as plt
from utils import *
from numpy import linalg
from conjugant_solver import *
import datetime
import pytz
import json


if __name__ == "__main__":

    #### First generate square mesh and add it to the global variables
    # Setup basic mesh parameters and declare them as global variables
    #Initialize all global variables
    x_max = 1
    y_max = 1
    number_x_points = 129
    number_y_points = 129
    dx = x_max / number_x_points
    dy = y_max / number_y_points

    v = 1/400

    # Solver Settings
    total_time = 15
    dt = .0025
    timesteps = total_time/dt
    print(timesteps)
    timesteps = int(timesteps)

    # Tolerances
    tol1 = 1e-3
    tol2 = 1e-4


    ## Assign values to each global variable
    # Mesh conditions
    nx = number_x_points
    ny = number_y_points
    nq = (nx-1)*ny + nx*(ny-1)
    n_p = nx*ny-1


    ## Boundary Wall velocities ##
    top_wall = (1, 0) # u, v velocity
    left_wall = (0, 0)
    right_wall = (0, 0)
    bottom_wall = (0 , 0)

    # CFL Number Calulation
    CFL = (top_wall[0]*dt/dx)
    print(f'CFL Number: {CFL}')

    #Array initializations
    # Create as 2D arrays that get stacked into a single vector q
    u_vel = np.zeros((nx,ny)) #Slightly larger than needed but makes iteration easier
    v_vel = np.zeros((nx,ny)) #Slightly larger than needed but makes iteration easier
    pressures = np.zeros((nx, ny)) #Slightly larger, one pressure should be pinned

     # Create Point arrays to map i, and j locations to q
    xu = np.zeros((nx, ny), dtype=int)
    xv = np.zeros((nx, ny), dtype=int)
    xp = np.zeros((nx, ny), dtype=int)

    for i in range(1,nx):
        for j in range(ny):
            xu[i,j] = i+j*(nx-1) - 1
    
    for i in range(nx):
        for j in range(ny-1):
            xv[i,j] = i+(j-1)*(nx) + (nx-1)*ny

    # Create lambda function to get index for given i ,j, assumes pinned pressure in 0,0
    for i in range(nx):
        for j in range(ny):
            if i !=0 or j!=0:
                xp[i,j] = i+j*nx -1

    # xu = lambda i,j: i+j*(nx-1) - 1
    # xv = lambda i,j: i+(j-1)*(nx) + (nx-1)*ny

    # # Add some u velocity
    # for i in range(len(u_vel[0])):
    #     for j in range(len(u_vel)):
    #         if i > (.2*nx) and i < (.8*nx):
    #             if j > (.2*ny) and j<(.8*ny):
    #                 u_vel[i,j] = 1

    # Initialize X and Y grid locations for velocities and presssures, doesn't include Boundary conditions
    x_array = np.linspace((dx/2), (x_max-dx/2), number_x_points)
    y_array = np.linspace((dy/2), (y_max-dy/2), number_y_points)

    # Pressure grid locations
    X_pressures, Y_pressures = np.meshgrid(x_array, y_array, indexing='ij')

    # U-Velocity grid locations
    # Shift coords Right
    x_array_u = x_array.copy()
    x_array_u[:] = [i+dx/2 for i in x_array_u]
    X_u, Y_u = np.meshgrid(x_array_u[:-1], y_array, indexing='ij')

    # V-Velocity grid locations
    # Shift coords Up
    y_array_v = y_array.copy()
    y_array_v[:] = [i+dy/2 for i in y_array_v]
    X_v, Y_v = np.meshgrid(x_array, y_array_v[:-1], indexing='ij')

    # Vorticity Locations for completeness
    X_w, Y_w = np.meshgrid(x_array_u[:-1], y_array_v[:-1], indexing='ij')

    # Grid corners for plotting collocated results
    x_corners = np.linspace(0, x_max, nx+1)
    y_corners = np.linspace(0, y_max, ny+1)
    X_corners, Y_corners = np.meshgrid(x_corners, y_corners, indexing='ij')

    ######### Lid Driven Cavity Flow Solver #########
    t_step = 0
    #t = 0
    Error = []
    # First pass initilizations and Static Boundary conditions
    A_old = advect(u_vel, v_vel, dx, dy, top_wall, left_wall, right_wall, bottom_wall)

    bc_laplace = dt*v*bc_lap(nx, ny, dx, dy, top_wall, left_wall, right_wall, bottom_wall)
    divergence_boundary = (1/dt)*bc_div(nx, ny, dx, dy, top_wall, left_wall, right_wall, bottom_wall)

    # Setup while loop to iterate over time
    #Counter on when to display time
    count = 0

    while t_step < timesteps:

        q = pack_q(u_vel, v_vel, nx, ny)
        p = pack_p(pressures, nx, ny)

        ##### First Fractional Step
        # First calulate the Right hand side of the first fractional step
        A_new = advect(u_vel, v_vel, dx, dy, top_wall, left_wall, right_wall, bottom_wall)
        s = S_times(u_vel, v_vel, dt, v, nx, ny, dx, dy)
        #laplace = lap(u_vel, v_vel, dx, dy)
        
        # Partial RHS terms
        full_advect = (dt/2)*(np.subtract(np.multiply(3,A_new), A_old))

        # Update non-linear advection term
        A_old = A_new

        # Full RHS
        RHS = np.subtract(s, full_advect)
        RHS = np.add(RHS, bc_laplace)
        
        if t_step == 0:
            # First solve u_F doesn't exist
            u_F = conjugant_solve1(q, RHS, tol1, dt, v, nx, ny, dx, dy)
        else:
            u_F = conjugant_solve1(u_F, RHS, tol1, dt, v, nx, ny, dx, dy)
        
        ###### Second Fractional Step
        u_F_u, u_F_v = unpack_q(u_F, nx, ny)

        divergence = (1/dt)*div(u_F_u, u_F_v, dx, dy)
        
        RHS2 = np.add(divergence, divergence_boundary)
        
        p = conjugant_solve2(p, RHS2, tol2, dt, v, nx, ny, dx, dy)

        ###### Third Step solve
        # Pressure term
        pressures = unpack_p(p, nx, ny)
        press_grad = gradient(pressures, dx, dy)
        u_grad, v_grad = unpack_q(press_grad, nx, ny)

        # R inverse term
        first = (dt*v/2)*lap(u_grad, v_grad, dx, dy)
        #second = np.power(first, 2)
        R_inverse = np.add(press_grad, first)
        #R_inverse = np.add(R_inverse, second)

        R_inverse = dt*R_inverse

        u_new = np.subtract(u_F, R_inverse)
        
        # Calculate Error
        top = abs(np.subtract(u_new, q))
        # per_point = np.divide(top, u_new)
        # Cahnge to really take residual since we are going to steady state solution
        Error.append(sum(abs(top)))

        # Repopulate arrays
        u_vel, v_vel = unpack_q(u_new, nx, ny)


        # Increment timestep
        t_step += 1
        #t += dt
        count += 1
        if count == 25:
            print(t_step/timesteps)
            count = 0
    


    # # Temp for troubleshooting
    #u_vel, v_vel = unpack_q(u_F, nx, ny)
    #print(u_vel, v_vel)

    # Collocate velocities
    U, V = collocate_velocity(u_vel, v_vel, nx, ny, top_wall, bottom_wall, right_wall, left_wall)

    #### Write all solved values to file
    stored_vals = {
                    'Reynolds':1/v,
                    'CFL':CFL,
                    'total time':total_time,
                    'dt': dt,
                    'nx':nx,
                    'ny':ny,
                    'dx':dx,
                    'dy':dy,
                    'x_max':x_max,
                    'y_max':y_max,
                    "u_vel":u_vel.tolist(),
                    'v_vel':v_vel.tolist(),
                    'U':U.tolist(),
                    'V':V.tolist(),
                    'pressures':pressures.tolist(),
                    'Error':Error 
                    }

    json_object = json.dumps(stored_vals, indent=4)
    current_time = datetime.datetime.now(pytz.timezone('US/Pacific'))
    time_name = f'{current_time.year}-{current_time.month}-{current_time.day}-{current_time.hour}-{current_time.minute}-{current_time.second}'

    with open(rf'C:\Users\Riley\Desktop\Code\FlowSim\Results\{time_name}_Re{1/v}_dx{dx}_dt{dt}_t{total_time}.json','w') as file:
        file.write(json_object)
        # file.write(f'nx = {nx} ny = {ny} dx = {dx} dy = {dy}\n')
        # file.write(f'U Velocities:\n')
        # file.write(f'{U}\n')
        # file.write(f'V Velocities:\n')
        # file.write(f'{V}\n')
        # file.write(f'Pressures:\n')
        # file.write(f'{pressures}\n')
        
        
        

    plt.figure
    plt.plot(Error)
    plt.title('Velocity Convergence')

    plt.figure()
    plt.quiver(X_corners, Y_corners, U, V)
    plt.xlim(0,x_max)
    plt.ylim(0,y_max)

    # plt.figure()
    # plt.contourf(X_u, Y_u, u_vel)
    # plt.colorbar()
    # plt.title("U Contour solved")

    plt.figure()
    plt.contourf(X_corners, Y_corners, U)
    plt.colorbar()
    plt.title("U Contour")

    # plt.figure()
    # plt.contourf(X_v, Y_v, v_vel[1::])
    # plt.colorbar()
    # plt.title("V Contour solved")

    plt.figure()
    plt.contourf(X_corners, Y_corners, V)
    plt.colorbar()
    plt.title("V contour")


    plt.show()
