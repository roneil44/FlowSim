#### A file that holds utility functions that are used is other documents

from scipy.interpolate import NearestNDInterpolator
import numpy as np

def collocate(field, current_x, current_y, new_x, new_y):
    '''This function linearly interpolates a field of values from one 
    set of grid coordinates to another.
    field = currnet values at current grid points
    current grid = array of current grid X,Y locations from np.meshgrid('ij')
    new_grid = array of new X,Y locations from np.meshgrid('ij')
    returns: new_field
    '''

    #Reformat data
    positions = list(zip(current_x.ravel(), current_y.ravel()))
    field = field.ravel()
    #print(positions)

    # print(np.shape(positions))
    # print(np.shape(field))
    
   
    interp = NearestNDInterpolator(positions, field.ravel())

    new_field = interp(new_x, new_y)

    return(new_field)

def pack_p(pressures, nx, ny):
    '''Take pressure array and pack it into a single list of size nx*ny'''
    p = np.zeros(nx*ny)
    index = 0
    
    for j in range(ny):
        for i in range(nx):
            p[index] = pressures[i,j]
            index += 1
    
    return p    
    


def pack_q(u_vels, v_vels, nx, ny):
    ''''Take and U and V velocity field and packs it into a single list of size (nx-1)*ny+nx*(ny-1)'''
    
    n_q = np.zeros((nx-1)*ny+nx*(ny-1))
    
    index = 0
    
    for i in range(1, nx):
        for j in range(ny):
            n_q[index] = u_vels[i,j]
            index+=1
            
    for i in range(nx):
        for j in range(1, ny):
            n_q[index] = v_vels[i,j]
            index+=1
            
    return(n_q)

def unpack_q(q_vels, nx, ny):
    ''' Takes the q velocity arrays and unpacks it into 2D vectors U and V
    Includes the boundary values defined in the initial definition of u_vel and v_vel'''
    
    # Repack it into 2d array for plotting
    U = np.zeros((nx, ny))
    V = np.zeros((nx, ny))
    
    index = 0

    # Build 2D U array
    for j in range(len(U[0])):
        for i in range(1, len(U)):
            U[i,j] = q_vels[index]
            index += 1

    # Build 2D V array
    for j in range(1, len(V[0])):
        for i in range(len(V)):
            V[i,j] = q_vels[index]
            index += 1
        
    return U, V

def unpack_p(p, nx, ny):
    '''Repack p back into a 2D array'''
    pressures = np.zeros((nx, ny))
    index = 0
    
    for j in range(ny):
        for i in range(nx):
            pressures[i,j] = p[index]
            index += 1
            
    return pressures
    
    

def collocate_velocity(u_vel, v_vel, nx, ny, top_wall, left_wall, right_wall, bottom_wall) -> list:
    '''Take the u and v velocity fields and interpolate them to cell corners + boundaries
    Returns a tuple that contains the (U, V) velocity arrays at each point
    '''
    U = np.zeros((nx+1, ny+1))
    V = np.zeros((nx+1, ny+1))
    
    # First populate the wall velocities
    # Bottom Row
    j=0
    for i in range(1, nx):
        U[i,j] = bottom_wall[0]
        V[i,j] = bottom_wall[1]
                    
    #Top Row
    j = ny
    for i in range(1,nx):
        U[i,j] = top_wall[0]
        V[i,j] = top_wall[1]
    
    #Left Row
    i = 0
    for j in range(1, ny):
        U[i,j] = left_wall[0]
        V[i,j] = left_wall[1]
    
    #Right Row
    i = nx
    for j in range(1, ny):
        U[i,j] = right_wall[0]
        V[i,j] = right_wall[1]
        
    #Bottom Left Corner
    U[0,0] = (bottom_wall[0] + left_wall[0])/2
    V[0,0] = (bottom_wall[1] + left_wall[1])/2
    
    #Top Left Corner
    U[0,ny] = (top_wall[0] + left_wall[0])/2
    V[0,ny] = (top_wall[1] + left_wall[1])/2
    
    #Top Right Corner
    U[nx,ny] = (top_wall[0] + right_wall[0])/2
    V[nx,ny] = (top_wall[1] + right_wall[1])/2
    
    #Bottom Right Corner
    U[nx,0] = (bottom_wall[0] + right_wall[0])/2
    V[nx,0] = (bottom_wall[1] + right_wall[1])/2 
        
    # Calculate middle points using interpolation
    
    for i in range(1,nx):
        for j in range(1,ny):
            U[i,j] = (u_vel[i,j-1] + u_vel[i,j])/2
            V[i,j] = (v_vel[i-1,j] + v_vel[i,j])/2
    
    return U, V

def collocate_pressure(pressure):
    '''interpolate pressures to the boundary of grid'''
    
    
    
    pass