##########
# This file implements a conjugant gradient solver
# Created by: Riley O'Neil
# Course: MAE250H
# Date: 5/10/20
##########

# Imports
import numpy as np
import numpy.typing as npt
from vector import *

def conjugant_solve(A:list[list], x0:list, b:list) -> list:
    ''' This function takes an initial guess x0, and the right hand side values
    b and solves Ax = b iteratively using the conjugant gradient method'''

    # Convert to numpy arrays to make vector operations easier
    A = np.array(A)
    x0 = np.array(x0)
    b= np.array(b)

    #Solver conditions
    max_iters = 1000
    i = 0
    eps = .01

    #First iteration
    r = b - Ax(A, x0)
    d = r
    
    delta_new = np.dot(r, r)
    delta_0 = delta_new


    while i<=max_iters and delta_new > eps*delta_0:
        
        q = Ax(A, d)
        alpha = delta_new / np.dot(d, q)
        x0 = x0 + alpha*d

        if i % 50 == 0:
            r = b - Ax(A, x0)
        else:
            r = r - alpha*q

        delta_old = delta_new
        delta_new = np.dot(r, r)
        beta = delta_new / delta_old

        d = r+beta*d

        i+=1
    return(x0)


def Ax(A:npt.ArrayLike, x:npt.ArrayLike) -> npt.NDArray:
    '''This function solves for A*x used in the conjugant gradient solver. For testing 
    the conjugant gradient solver a full A matrix is defined. In the future this can 
    be simplified an all sparse values can be removed'''


    b = np.matvec(A, x)

    return b

def Ru(u, dt, v):
    '''This function computes R*u for the first step of the fractional step 
    projection method'''
    
    R = u + dt*(v/2)*(lap(u))

    pass

def Dp(GP):
    '''This function computes D*R^(-1)GP for the second half of the fractional
    step projection method'''
