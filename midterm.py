######
# Main file for incompressible flow solver
# #######



## Imports
from vector import *

import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import math
from utils import *
from numpy import linalg

if __name__ == "__main__":
    #### First generate square mesh and add it to the global variables
    # Setup basic mesh parameters and declar them as global variables
    #Initilize all global variables
    x_max = 1
    y_max = 1
    number_x_points = 20
    number_y_points = 20
    dx = x_max / number_x_points
    dy = y_max / number_y_points

    ## Assign values to each global variable
    # Mesh conditions
    nx = number_x_points
    ny = number_y_points
    nq = (nx-1)*ny + nx*(ny-1)
    n_p = nx*ny-1

    ## Boundary Wall velocities ##
    top_wall = (0, 0) # u, v velocity
    left_wall = (0, 0)
    right_wall = (0, 0)
    bottom_Wall = (0 , 0)

    #Array initilizations
    # Create as 2D arrays that get stacked into a single vector q
    u_vel = np.zeros((nx,ny)) #Slightly larger than needed but makes iteration easier
    v_vel = np.zeros((nx,ny)) #Slightly larger than needed but makes iteration easier
    pressures = np.zeros((nx, ny)) #Slightly larger, one pressure should be pinned

     # Create Point arrays to map i, and j locations to q
    xu = np.zeros((nx, ny), dtype=int)
    xv = np.zeros((nx, ny), dtype=int)
    xp = np.zeros((nx, ny), dtype=int)

    index = 0
    for j in range(ny):
        for i in range(1,nx):
            xu[i,j] = index
            index +=1

    # for i in range(1,nx):
    #     for j in range(ny):
    #         xu[i,j] = i+j*(nx-1) - 1
    
    for i in range(nx):
        for j in range(1,ny):
            xv[i,j] = i+(j-1)*(nx) + (nx-1)*ny

    # Create xp assuming pinned pressure in 0,0
    # for i in range(nx):
    #     for j in range(ny):
    #         if i !=0 or j!=0:
    #             xp[i,j] = i*ny+j -1

    index = 0
    for j in range(ny):
        for i in range(nx):
            if i !=0 or j!=0:
                xp[i,j] = index
                index+=1

    q = pack_q(u_vel, v_vel, nx, ny)

    # print(xu[2,3])
    # print(xp)

    # Initialize X and Y grid locations for velocities and presssures, doesn't include Boundary conditions
    x_array = np.linspace((dx/2), (x_max-dx/2), number_x_points)
    y_array = np.linspace((dy/2), (y_max-dy/2), number_y_points)

    # Presure grid locations
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


    #####
    # Midterm compare vector operations



    ######## Analytic analysis of Vector operations #########

    # Analytic comparison using sin functions
    n = 2
    m = 3

    ### SETUP Assign values to u and v arrays ###
    #Since we slightly oversized our arrays u_vel iterators need to be slightly adjusted
    for i in range(len(X_u)):
        for j in range(len(X_u[0])):
            u_vel[i+1,j] = math.sin(m*math.pi*X_u[i,j]/x_max)*math.sin(n*math.pi*Y_u[i,j]/y_max)

    for i in range(len(X_v)):
        for j in range(len(X_v[0])):
            v_vel[i,j+1] = math.sin(n*math.pi*X_v[i,j]/x_max)*math.sin(m*math.pi*Y_v[i,j]/y_max)
            
    # Assign values to the Pressure array based on a similar sin function
    for i in range(len(X_pressures)):
        for j in range(len(X_pressures[0])):
            pressures[i,j] = math.sin(m*math.pi*X_pressures[i,j]/x_max)*math.sin(n*math.pi*Y_pressures[i,j]/y_max)


    ### Gradient Check ####

    # Initialize the two vector components of the gradient
    grad_u = np.zeros((nx-1, ny))
    grad_v = np.zeros((nx, ny-1))

    ## Solve exact for u
    for i in range(len(grad_u)):
        for j in range(len(grad_u[0])):
            grad_u[i,j] = (m*math.pi/x_max)*math.cos(m*math.pi*X_pressures[i,j]/x_max)*math.sin(n*math.pi*Y_pressures[i,j]/y_max)
            
    ## Solve exact for v
    for i in range(len(grad_v)):
        for j in range(len(grad_v[0])):
            grad_v[i,j] = (n*math.pi/y_max)*math.sin(m*math.pi*X_pressures[i,j]/x_max)*math.cos(n*math.pi*Y_pressures[i,j]/y_max)

    ## Get numeric solution
    press = pack_p(pressures, nx, ny)
    num_grad = gradient(pressures, dx, dy)
    
    # Tryin to fix error with new gradietn operator
    #num_grad = gradient2(pressures, dx, dy)

    # print(xp)
    # print(xu)
    # print(xv)
    # print(num_grad)
    # #print(num_grad2)
    # print(grad_u)


    # # Repack it into 2d array for plotting
    num_grad_array_u = np.zeros((len(x_array_u)-1, len(y_array)))
    num_grad_array_v = np.zeros((len(x_array), len(y_array_v)-1))
    index = 0

    # Build 2D U array
    for j in range(len(num_grad_array_u[0])):
        for i in range(len(num_grad_array_u)):
            if index != 0:
                num_grad_array_u[i,j] = num_grad[index]
            index += 1

    # Build 2D V array
    for j in range(len(num_grad_array_v[0])):
        for i in range(len(num_grad_array_v)):
            num_grad_array_v[i,j] = num_grad[index]
            index += 1

    # num_grad_array_u, num_grad_array_v = unpack_q(num_grad, nx, ny)

    # Remove extra points
    # num_u_short = 
            
    ## compute error

    grad_u_error = np.subtract(grad_u, num_grad_array_u)
    grad_v_error = np.subtract(grad_v, num_grad_array_v)

    # Square for L2 error
    grad_u_error = grad_u_error*grad_u_error
    grad_v_error = grad_v_error *grad_v_error 

    grad_u_avg_error = np.sum(grad_u_error) / (len(grad_u_error) * len(grad_u_error[0]))
    grad_v_avg_error = np.sum(grad_v_error) / (len(grad_v_error) * len(grad_v_error[0]))
        

    ### Divergence Check ####

    # Check divergence
    exact_divergence = np.zeros((nx, ny))

    for i in range(len(X_pressures)):
        for j in range(len(X_pressures[0])):
            exact_divergence[i,j] = ((m*math.pi/x_max)*math.cos(m*math.pi*X_pressures[i,j]/x_max)*math.sin(n*math.pi*Y_pressures[i,j]/y_max) + 
            (m*math.pi/y_max)*math.sin(n*math.pi*X_pressures[i,j]/x_max)*math.cos(m*math.pi*Y_pressures[i,j]/y_max))

    # Compute numeric divergence
    numeric_divergence = div(u_vel, v_vel, dx, dy)

    # Repack it into 2d array for plotting
    num_div_array = np.zeros((nx, ny))
    index = 0

    for j in range(len(X_pressures[0])):
        for i in range(len(X_pressures)):
            if i != 0 or j != 0:
                num_div_array[i,j] = numeric_divergence[index]
                index += 1

    # Calculate divergence error
    div_error = np.absolute(np.subtract(exact_divergence, num_div_array))
    div_error_sum = np.sum(div_error)
    div_error_avg = div_error_sum/(len(div_error) * len(div_error[0]))
    ####

    ### Laplacian Check ####

    # Initialize the two vector components of the laplacian
    laplace_u = np.zeros((nx-1, ny))
    laplace_v = np.zeros((nx, ny-1))

    # print(np.shape(laplace_u))
    # print(np.shape(X_u))

    # print(np.shape(laplace_v))
    # print(np.shape(X_v))
    # print(Y_u[1,1])

    ## Solve exact for u
    for i in range(len(laplace_u)):
        for j in range(len(laplace_u[0])):
            rh = math.sin(m*math.pi*(X_u[i,j])/x_max)*math.sin(n*math.pi*Y_u[i,j]/y_max)
            laplace_u[i,j] = -(m**2*math.pi**2/x_max**2)*rh - (n**2*math.pi**2/y_max**2)*rh
    ## Solve exact for v
    for i in range(len(laplace_v)):
        for j in range(len(laplace_v[0])):
            rh = math.sin(n*math.pi*X_v[i,j]/x_max)*math.sin(m*math.pi*Y_v[i,j]/y_max)
            laplace_v[i,j] = -(n**2*math.pi**2/x_max**2)*rh - (m**2*math.pi**2/y_max**2)*rh

    ## Get numeric solution
    num_lap = lap(u_vel, v_vel, dx, dy)

    # Repack it into 2d array for plotting
    num_lap_array_u = np.zeros((len(x_array_u)-1, len(y_array)))
    num_lap_array_v = np.zeros((len(x_array), len(y_array_v)-1))
    index = 0

    # Build 2D U array
    for j in range(len(num_lap_array_u[0])):
        for i in range(len(num_lap_array_u)):
            num_lap_array_u[i,j] = num_lap[index]
            index += 1

    # Build 2D V array
    for j in range(len(num_lap_array_v[0])):
        for i in range(len(num_lap_array_v)):
            num_lap_array_v[i,j] = num_lap[index]
            index += 1

    # # Co-locate U and V arrays on pressure centers
    co_lap_u = collocate(laplace_u, X_u, Y_u, X_pressures, Y_pressures)
    co_lap_v = collocate(laplace_v, X_v, Y_u, X_pressures, Y_pressures)

    co_lap_u_num = collocate(num_lap_array_u, X_u, Y_u, X_pressures, Y_pressures)
    co_lap_v_num = collocate(num_lap_array_v, X_v, Y_v, X_pressures, Y_pressures)

    # calculate error 
    # print(np.shape(co_lap_u))
    # print(np.shape(co_lap_u_num))
    # print(np.shape(X_u))
    lap_u_error = np.absolute(np.subtract(laplace_u, num_lap_array_u))
    lap_v_error = np.absolute(np.subtract(laplace_v, num_lap_array_v))

    lap_u_avg_error = np.sum(lap_u_error) / (len(lap_u_error) * len(lap_u_error[0]))
    lap_v_avg_error = np.sum(lap_v_error) / (len(lap_v_error) * len(lap_v_error[0]))

    #######

    #### Non Linear Advection Check ####
    # Initialize arrays
    adv_u = np.zeros((nx-1, ny))
    adv_v = np.zeros((nx, ny-1))

    # Change the u and v functions to make solving by hand easier also make sure U = 1 at top boundary since our 
    # solver explicitly includes Boundary conditions

    for i in range(len(X_u)):
        for j in range(len(X_u[0])):
            u_vel[i+1,j] = math.sin(m*math.pi*X_u[i,j]/(x_max))

    for i in range(len(X_v)):
        for j in range(len(X_v[0])):
            v_vel[i,j+1] = math.sin(n*math.pi*Y_v[i,j]/y_max)


    ## Solve exact for Nx from Analytic Expression
    for i in range(len(adv_u)):
        for j in range(len(adv_v)):

            adv_u[i,j] = (  (2*m*math.pi/x_max)*math.sin(m*math.pi*X_u[i,j]/x_max) * 
                        math.cos(m*math.pi*X_u[i,j]/x_max) + (n*math.pi/y_max)*math.sin(m*math.pi*X_u[i,j]/x_max)*
                        math.cos(n*math.pi*Y_u[i,j]/y_max)      )

    ## Solve exact for Ny
    for i in range(len(laplace_v)):
        for j in range(len(laplace_v[0])):

            adv_v[i,j] = (  (2*n*math.pi/y_max)*math.sin(n*math.pi*Y_v[i,j]/y_max) * 
                        math.cos(n*math.pi*Y_v[i,j]/y_max) + (m*math.pi/x_max)*math.sin(n*math.pi*Y_v[i,j]/y_max)*
                        math.cos(m*math.pi*X_v[i,j]/x_max)      )


    ## Get numeric solution
    num_adv = advect(u_vel, v_vel, dx, dy, top_wall, left_wall, right_wall, bottom_Wall)


    # Repack it into 2d array for plotting
    num_adv_array_u = np.zeros((len(x_array_u)-1, len(y_array)))
    num_adv_array_v = np.zeros((len(x_array), len(y_array_v)-1))
    index = 0

    # Build 2D U array
    for j in range(len(num_adv_array_u[0])):
        for i in range(len(num_adv_array_u)):
            num_adv_array_u[i,j] = num_adv[index]
            index += 1

    # Build 2D V array
    for j in range(len(num_adv_array_v[0])):
        for i in range(len(num_adv_array_v)):
            num_adv_array_v[i,j] = num_adv[index]
            index += 1

    # Compute Advection Error
    adv_u_error = np.absolute(np.subtract(adv_u, num_adv_array_u))
    adv_v_error = np.absolute(np.subtract(adv_v, num_adv_array_v))

    adv_u_avg_error = np.sum(adv_u_error) / (len(adv_u_error) * len(adv_u_error[0]))
    adv_v_avg_error = np.sum(adv_v_error) / (len(adv_v_error) * len(adv_v_error[0]))


    ###### ########





    ##### Plotting Midterm #####

    #### Gradient Plots ####

    #Exact Gradient
    # plt.figure()
    # plt.contourf(X_u, Y_u, grad_u)
    # plt.title('Exact Gradient U-Component')
    # plt.colorbar(label='Gradient')
    # plt.xlim((0,x_max))
    # plt.ylim((0,y_max))

    # #Numeric Gradient
    # plt.figure()
    # plt.contourf(X_u, Y_u, num_grad_array_u)
    # plt.title('Numeric Gradient U-Component')
    # plt.colorbar(label='Gradient')
    # plt.xlim((0,x_max))
    # plt.ylim((0,y_max))

    # #Exact V
    # plt.figure()
    # plt.contourf(X_v, Y_v, grad_v)
    # plt.title('Exact Gradient V-Component')
    # plt.colorbar(label='Gradient')
    # plt.xlim((0,x_max))
    # plt.ylim((0,y_max))

    # #Numeric V
    # plt.figure()
    # plt.contourf(X_v, Y_v, num_grad_array_v)
    # plt.title('Numeric Gradient V-Component')
    # plt.colorbar(label='Gradient')
    # plt.xlim((0,x_max))
    # plt.ylim((0,y_max))

    # # #Gradient U Error
    # plt.figure()
    # plt.contourf(X_u, Y_u, grad_u_error, norm=matplotlib.colors.LogNorm())
    # plt.title('Gradient U-Component Error')
    # plt.colorbar(label='Error')
    # plt.annotate(f'dx = {dx}\ndy = {dy}\navg point error = {round(grad_u_avg_error, 5)}', (.1*x_max, .1*y_max))
    # plt.xlim((0,x_max))
    # plt.ylim((0,y_max))

    # #Gradient V Error
    # plt.figure()
    # plt.contourf(X_v, Y_v, grad_v_error, norm=matplotlib.colors.LogNorm())
    # plt.title('Gradient V-Component Error')
    # plt.colorbar(label='Error')
    # plt.annotate(f'dx = {dx}\ndy = {dy}\navg point error = {round(grad_v_avg_error, 5)}', (.1*x_max, .1*y_max))
    # plt.xlim((0,x_max))
    # plt.ylim((0,y_max))

    # # Velocity Plots ######
    # plt.figure()
    # plt.contourf(X_u, Y_u, u_vel[1:])
    # plt.title('U-velocity')
    # plt.colorbar(label='Velocity')
    # plt.xlim((0,x_max))
    # plt.ylim((0,y_max))

    # # V Plot
    # plt.figure()
    # plt.contourf(X_v, Y_v, [sublist[1:] for sublist in v_vel])
    # plt.title('V-velocity')
    # plt.colorbar(label='Velocity')
    # plt.xlim((0,x_max))
    # plt.ylim((0,y_max))

    # #### Divergence Plots #####
    # #Divergence plot
    # plt.figure()
    # plt.contourf(X_pressures, Y_pressures, exact_divergence)
    # plt.title('Exact Divergence')
    # plt.colorbar(label='Divergence')
    # plt.xlim((0,x_max))
    # plt.ylim((0,y_max))

    # # Numeric Divergence
    # plt.figure()
    # plt.contourf(X_pressures, Y_pressures, num_div_array)
    # plt.title('Numeric Divergence')
    # plt.colorbar(label='Divergence')
    # plt.xlim((0,x_max))
    # plt.ylim((0,y_max))

    # # Divergence Error
    # plt.figure()
    # plt.contourf(X_pressures, Y_pressures, div_error, norm=matplotlib.colors.LogNorm())
    # plt.title('Divergence Error')
    # plt.colorbar(label='Error')
    # plt.annotate(f'dx = {dx}\ndy = {dy}\navg point error = {round(div_error_avg, 5)}', (.1*x_max, .1*y_max))
    # plt.xlim((0,x_max))
    # plt.ylim((0,y_max))
    # # print(f'u_vel 1,0:{u_vel[1][0]/dx}')
    # # print(f'v_vel 0,1:{v_vel[0][1]/dy}')
    # # print(f'expected:{u_vel[1][0]/dx + v_vel[0][1]/dy}')
    # print(f'exact: {exact_divergence[0][0]}')
    # print(f'numeric: {num_div_array[0][0]}')
    # print(bc_div(nx, ny, dx, dy, top_wall, left_wall, right_wall, bottom_Wall))

    # # #### Laplacian Plots #####

    # #Laplacian U plots
    plt.figure()
    plt.contourf(X_u, Y_u, laplace_u)
    plt.title('Exact Laplacian U-Component')
    plt.colorbar(label='Laplace')
    plt.xlim((0,x_max))
    plt.ylim((0,y_max))

    plt.figure()
    plt.contourf(X_u, Y_u, num_lap_array_u)
    plt.title('Numeric Laplacian U-Component')
    plt.colorbar(label='Laplace')
    plt.xlim((0,x_max))
    plt.ylim((0,y_max))

    # Laplacian V plots
    plt.figure()
    plt.contourf(X_v, Y_v, laplace_v)
    plt.title('Exact Laplacian V-Component')
    plt.colorbar(label='Laplace')
    plt.xlim((0,x_max))
    plt.ylim((0,y_max))

    plt.figure()
    plt.contourf(X_v, Y_v, num_lap_array_v)
    plt.title('Numeric Laplacian V-Component')
    plt.colorbar(label='Laplace')
    plt.xlim((0,x_max))
    plt.ylim((0,y_max))

    # # #print(np.shape(Y_pressures))
    # # plt.figure()
    # # plt.quiver(X_pressures, Y_pressures, co_lap_u, co_lap_v)
    # # plt.title('Laplacian Quiver Exact')
    # # plt.xlim((0,x_max))
    # # plt.ylim((0,y_max))

    # # plt.figure()
    # # plt.quiver(X_pressures, Y_pressures, co_lap_u_num, co_lap_v_num)
    # # plt.title('Laplacian Quiver Numeric')
    # # plt.xlim((0,x_max))
    # # plt.ylim((0,y_max))

    plt.figure()
    plt.contourf(X_u, Y_u, lap_u_error, norm=matplotlib.colors.LogNorm())
    plt.title('Laplacian U-Component Error')
    plt.colorbar(label='Error')
    plt.annotate(f'dx = {dx}\ndy = {dy}\navg point error = {round(lap_u_avg_error, 5)}', (.1*x_max, .1*y_max))
    plt.xlim((0,x_max))
    plt.ylim((0,y_max))

    plt.figure()
    plt.contourf(X_v, Y_v, lap_v_error, norm=matplotlib.colors.LogNorm())
    plt.title('Laplacian V-Component Error')
    plt.colorbar(label='Error')
    plt.annotate(f'dx = {dx}\ndy = {dy}\navg point error = {round(lap_v_avg_error, 5)}', (.1*x_max, .1*y_max))
    plt.xlim((0,x_max))
    plt.ylim((0,y_max))

    # # ### Non-Linear Advection Plots ####

    # #Numeric U
    # plt.figure()
    # plt.contourf(X_u, Y_u, num_adv_array_u)
    # plt.title('Numeric Advection U-Component')
    # plt.colorbar(label='Advection')
    # plt.xlim((0,x_max))
    # plt.ylim((0,y_max))

    # #Exact U
    # plt.figure()
    # plt.contourf(X_u, Y_u, adv_u)
    # plt.title('Exact Advection U-Component')
    # plt.colorbar(label='Advection')
    # plt.xlim((0,x_max))
    # plt.ylim((0,y_max))

    # # Numeric V
    # plt.figure()
    # plt.contourf(X_v, Y_v, num_adv_array_v)
    # plt.title('Numeric Advection V-Component')
    # plt.colorbar(label='Advection')
    # plt.xlim((0,x_max))
    # plt.ylim((0,y_max))

    # #Exact V
    # plt.figure()
    # plt.contourf(X_v, Y_v, adv_v)
    # plt.title('Exact Advection V-Component')
    # plt.colorbar(label='Advection')
    # plt.xlim((0,x_max))
    # plt.ylim((0,y_max))

    # #U Error
    # plt.figure()
    # plt.contourf(X_u, Y_u, adv_u_error, norm=matplotlib.colors.LogNorm())
    # plt.title('Advection U-Component Error')
    # plt.colorbar(label='Error')
    # plt.annotate(f'dx = {dx}\ndy = {dy}\navg point error = {round(adv_u_avg_error, 5)}', (.1*x_max, .1*y_max))
    # plt.xlim((0,x_max))
    # plt.ylim((0,y_max))

    # #V Error
    # plt.figure()
    # plt.contourf(X_v, Y_v, adv_v_error, norm=matplotlib.colors.LogNorm())
    # plt.title('Advection V-Component Error')
    # plt.colorbar(label='Error')
    # plt.annotate(f'dx = {dx}\ndy = {dy}\navg point error = {round(adv_v_avg_error, 5)}', (.1*x_max, .1*y_max))
    # plt.xlim((0,x_max))
    # plt.ylim((0,y_max))

    # ###Plotting staggered grid######
    # plt.figure()
    # plt.title('Staggered Grid Formulation')
    # plt.scatter(X_pressures, Y_pressures, c='black', marker='*')
    # plt.scatter(X_u, Y_u, c='blue', marker='>')
    # plt.scatter(X_v, Y_v, c='blue', marker='^')
    # plt.scatter(X_w, Y_w, c='red', marker='o')
    # plt.legend(['Pressure', 'U-velocity','V-velocity', 'Vorticity'],loc=1)
    # plt.xticks(np.linspace(0,x_max, number_x_points+1))
    # plt.yticks(np.linspace(0,y_max,number_y_points+1))
    # plt.grid(True, alpha=.2)
    # plt.xlim(0,x_max)
    # plt.ylim(0, y_max)
    # plt.show()


    plt.show()


    ##### PLOTTING FOR REPORTS #####




    ########### BACKUP TO BE REMOVED #############


    # First compute the gradient of a known function
    # Using the pressure array since it already exists
    # for i in range(len(pressures)):
    #     for j in range(len(pressures[0])):
    #         pressures[i, j] = (2*i)+j
    # #print(pressures)

    # pressure_gradient = gradient(pressures, dx, dy)
    # #print(pressure_gradient)

    # plt.figure()
    # plt.contourf(X_pressures, Y_pressures, pressures)
    # plt.colorbar()
    # # plt.show()


    # Next compute divergence of known function
    # for i in range(len(u_vel)):
    #     for j in range(len(u_vel[0])):
    #         u_vel[i,j] = j
    #         v_vel[i,j] = 0
    # divergences = div(u_vel, v_vel, dx, dy)
    # print(u_vel)
    # print(divergences)


    # #v_vel = np.zeros((nx,ny))

    # # Compute Laplacian of velocity arrays
    # laplacian = lap(u_vel, v_vel, dx, dy)
    # #print(laplacian)

    # # Compute Non-linear advection
    # advection = advect(u_vel, v_vel, dx, dy, top_wall, left_wall, right_wall, bottom_Wall)
    # print(advection)
