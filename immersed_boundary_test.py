##### This script holds functions that assist in the development of an immersed boundary method

import numpy as np
import math
from immersed_boundary import *
from matplotlib import pyplot as plt
from utils import *

#Initialize all global variables
x_max = 1
y_max = 1
number_x_points = 13
number_y_points = 13
dx = x_max / number_x_points
dy = y_max / number_y_points
ds = (dx+dy)/2
#print(ds)
 ## Assign values to each global variable
# Mesh conditions
nx = number_x_points
ny = number_y_points
nq = (nx-1)*ny + nx*(ny-1)
n_p = nx*ny-1

# Initialize X and Y grid locations for velocities and presssures, doesn't include Boundary conditions
x_array = np.linspace((dx/2), (x_max-dx/2), number_x_points)
y_array = np.linspace((dy/2), (y_max-dy/2), number_y_points)

x_array_u = x_array.copy()
x_array_u[:] = [i+dx/2 for i in x_array_u]
X_u, Y_u = np.meshgrid(x_array_u[:-1], y_array, indexing='ij')

# V-Velocity grid locations
# Shift coords Up
y_array_v = y_array.copy()
y_array_v[:] = [i+dy/2 for i in y_array_v]
X_v, Y_v = np.meshgrid(x_array, y_array_v[:-1], indexing='ij')

##### Test Get circle coordinates function
x_c, y_c = get_points_on_circle(.275, .5, .5, ds)

## Calulate relationship between lagragian and cartesian points
cart_x, cart_y = calc_influence(X_u.flatten(order='F'), Y_u.flatten(order='F'), x_c, y_c, ds)
cart_x2, cart_y2 = calc_influence(X_v.flatten(order='F'), Y_v.flatten(order='F'), x_c, y_c, ds)
x_points = [tup[0] for tup in cart_x]
y_points = [tup[0] for tup in cart_y]
x_points2 = [tup[0] for tup in cart_x2]
y_points2 = [tup[0] for tup in cart_y2]

### Test Eu function

u_vels = np.ones((nx,ny))
v_vels = np.zeros((nx, ny))



x_loc_u, y_loc_u = np.meshgrid(np.linspace(0,x_max-dx,nx), np.linspace(dy/2,y_max-dy/2,ny), indexing='ij')
x_loc_v, y_loc_v = np.meshgrid(np.linspace(dx/2,x_max-dx/2,nx), np.linspace(0,y_max-dy,ny), indexing='ij')
# print(np.shape(x_loc_u.flatten(order='F')))
# print(np.shape(y_loc_u.flatten(order='F')))

#Apply gradient of u_vel
for i in range(nx):
    for j in range(ny):
        u_vels[i,j] = i*dx*2

# Get the transforms between the coordinate systems
coord_u = calc_q(x_loc_u.flatten(order='F'), y_loc_u.flatten(order='F'), x_c, y_c, ds)
coord_v = calc_q(x_loc_u.flatten(order='F'), y_loc_u.flatten(order='F'), x_c, y_c, ds)

# if the coordinate is near the first point on circle show it
# Get the index of every point near the first point
l_c_u = []
for coords in coord_u:
    if coords[0] == 0:
        l_c_u.append(coords)

l_c_v = []
for coords in coord_v:
    if coords[0] == 0:
        l_c_v.append(coords)

l_c_u1 = []
for coords in coord_u:
    if coords[0] == 1:
        l_c_u1.append(coords)

l_c_v1 = []
for coords in coord_v:
    if coords[0] == 1:
        l_c_v1.append(coords)

print(f'x_loc_u {np.size(x_loc_u)}')
print(f'l_c_v {l_c_v}')
X_u_flat = x_loc_u.flatten(order='F')
Y_u_flat = y_loc_u.flatten(order='F')
X_v_flat = x_loc_v.flatten(order='F')
Y_v_flat = y_loc_v.flatten(order='F')
# Translate that index actual points that can be plotted
cart_xu0 = [X_u_flat[point[1]] for point in l_c_u]
cart_yu0 = [Y_u_flat[point[1]] for point in l_c_u]
cart_xv0 = [X_v_flat[point[1]] for point in l_c_v]
cart_yv0 = [Y_v_flat[point[1]] for point in l_c_v]

print(f'cart_xv0 {cart_xv0}')

cart_xu1 = [X_u_flat[point[1]] for point in l_c_u1]
cart_yu1 = [Y_u_flat[point[1]] for point in l_c_u1]
cart_xv1 = [X_v_flat[point[1]] for point in l_c_v1]
cart_yv1 = [Y_v_flat[point[1]] for point in l_c_v1]


b_vels_u = Eu(u_vels.flatten(order='F'), coord_u, ds)
b_vels_v = Eu(v_vels.flatten(order='F'), coord_v, ds)

# b_vels_mat_u = np.zeros((len(x_c), len(y_c)))

# print(f'len b_vels_u {len(b_vels_u)}')
# print(f'len x_c {len(x_c)}')
# index = 0
# for j in range(len(y_c)):
#     for i in range(len(x_c)):
#         b_vels_mat_u[i,j] = b_vels_u[index]
#         index+=1

#b_vels_v = Eu(v_vels.flatten(order='F'), coord_v, ds)
# print(x_c)
# print(y_c)
# print(coord_u)
#print(b_vels_u)
#print(b_vels_v)

### Test H Function

F_x = np.ones(len(x_c))
F_y = np.ones(len(x_c))

H_u = HF(F_x, coord_u, ds, (nx)*ny)
H_v = HF(F_y, coord_v, ds, (ny)*nx)

#print(H_u)

H = np.append(H_u, H_v)
print(f'H= {np.shape(H)}')
h_x, h_y = unpack_q2(H, nx, ny)
print(f'Hx= {np.shape(h_x)}')

## For plotting remove extra u_vels
u_vels = u_vels[1:]#
v_vels = [row[1:] for row in v_vels]

plt.figure()
plt.scatter(x_c, y_c)
plt.scatter(x_points, y_points, color='red')
plt.scatter(x_points2, y_points2, color='purple')
plt.title('3 Grid Cell Delta Coverage')
plt.legend(['Lagragian Boundary Points','Cartesian U points with 1.5*ds','Cartesian V points with 1.5*ds'])
plt.xlim(0,x_max)
plt.annotate(f'Grid {nx}x{ny}', (.05,.1))
plt.ylim(0,y_max)

plt.figure()
plt.scatter(x_loc_u,y_loc_u,color='red')
plt.scatter(x_loc_v,y_loc_v,color='purple')
plt.legend(['Cartesian U','Cartesian V'])
plt.xlim(0,x_max)
plt.annotate(f'Grid {nx}x{ny}', (.05,.1))
plt.ylim(0,y_max)

plt.figure()
plt.scatter(x_c[0], y_c[0])
plt.scatter(cart_xu0, cart_yu0, color='red')
plt.scatter(cart_xv0, cart_yv0, color='purple')
plt.legend(['Lagragian Boundary','Cartesian U','Cartesian V'])
plt.xlim(0,x_max)
plt.ylim(0,y_max)
plt.annotate(f'Grid {nx}x{ny}', (.05,.1))
plt.title('Points Related to a Single Point')

plt.figure()
plt.scatter(x_c[1], y_c[1])
plt.scatter(cart_xu1, cart_yu1, color='red')
plt.scatter(cart_xv1, cart_yv1, color='purple')
plt.legend(['Lagragian Boundary','Cartesian U','Cartesian V'])
plt.xlim(0,x_max)
plt.ylim(0,y_max)
plt.annotate(f'Grid {nx}x{ny}', (.05,.1))
plt.title('Points Related to a Single Point')

plt.figure()
plt.contourf(x_loc_u, y_loc_u, h_x)
plt.title('H_u')
plt.xlim(0,x_max)
plt.ylim(0,y_max)
plt.annotate(f'Grid {nx}x{ny}', (.05,.1))
plt.colorbar()

plt.figure()
plt.contourf(x_loc_v, y_loc_v, h_y)
plt.title('H_v')
plt.xlim(0,x_max)
plt.ylim(0,y_max)
plt.annotate(f'Grid {nx}x{ny}', (.1,.1))
plt.colorbar()

plt.figure()
plt.contourf(X_u, Y_u, u_vels)
plt.scatter(x_c, y_c, c=b_vels_u)
plt.title('u_vel Contour')
plt.xlim(0,x_max)
plt.ylim(0,y_max)
plt.annotate(f'Grid {nx}x{ny}', (.1,.1))
plt.colorbar()


# plt.figure()
# plt.contourf(x_c, y_c, b_vels_mat_u)
# plt.title('U_vels on Boundary')
# plt.xlim(0,x_max)
# plt.ylim(0,y_max)



plt.show()