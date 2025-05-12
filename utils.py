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
