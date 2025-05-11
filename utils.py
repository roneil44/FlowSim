#### A file that holds utility functions that are used is other documents

from scipy.interpolate import RegularGridInterpolator

def collocate(field, current_x, current_y, new_x, new_y):
    '''This function linearly interpolates a field of values from one 
    set of grid coordinates to another.
    field = currnet values at current grid points
    current grid = array of current grid X,Y locations from np.meshgrid('ij')
    new_grid = array of new X,Y locations from np.meshgrid('ij')
    returns: new_field
    '''

    interp = RegularGridInterpolator((current_x, current_y), field, bounds_error=False,fill_value=None)

    new_field = interp((new_x, new_y))

    return(new_field)