########
# Used to test conjugant gradient solver
# Riley O'Neil
#
########
import numpy as np
from matplotlib import pyplot as plt
from utils import *
from conjugant_solver import *

#Initializations
x0 = [2,1]
b = [3,2]
A = [[5,1],[1,8]]

nx = 3
ny = 3
v = 1
dt = .5
dx = 1/nx
dy = 1/ny

x_points = np.linspace(0,1,nx+1)
y_points = np.linspace(0,1,ny+1)
co_x, co_y = np.meshgrid(x_points, y_points,indexing='ij')

u_vels = np.ones((nx, ny))
v_vels = np.ones((nx,ny))
top_wall = (1,0)
bottom_wall = (0,0)
left_wall = (0,0)
right_wall = (0,0)



###### Test basic Conjugant solver
#Test Ax
print(Ax(A, x0))

# Run conjugant solve
solution = conjugant_solve(A, x0, b)
print(solution)

# Direct solve with numpy matrix solver
real = np.linalg.solve(A, b)
print(real)

# Test with larger matrix
# First generate symmetric positive matrix
A1 = [[5, 2, 1, 7],[1, 3, 8, 4],[5, 2, 9, 3],[4, 2, 6, 2]]
A2 = np.transpose(A1)
A = np.matmul(A1, A2)
print(A)

# Solution vector
b = [1, 4, 6, 7]

#Initial guess
x0 = [0, 0, 0, 0]

#Conjugant gradient solve
solution= conjugant_solve(A, x0 ,b)
print(solution)

#Direct solve using numpy library
real = np.linalg.solve(A, b)
print(real)

##########

###### Test Ru
# q = pack_q(u_vels, v_vels, nx, ny)

# print(q)

# ru_calc = Ru(q, dt, v, nx, ny, dx, dy)

# print(ru_calc)