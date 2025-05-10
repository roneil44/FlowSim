##########
# This file implements a conjugant gradient solver
# Created by: Riley O'Neil
# Course: MAE250H
# Date: 5/10/20
##########

# Imports
import numpy as np

def conjugant_solve(A:list[list], x0:list, b:list) -> list:
    ''' This function takes an initial guess x0, and the right hand side values
    b and solves Ax = b iteratively using the conjugant gradient method'''

    pass


def Ax(A:list[list], x:list) -> list:
    '''This function solves for A*x used in the conjugant gradient solver. For testing 
    the conjugant gradient solver a full A matrix is defined. In the future this can 
    be simplified an all sparse values can be removed'''


    b = np.matvec(A, x)

    return b

