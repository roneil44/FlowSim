########
# File to test functions in utils.py


#######
import numpy as np
from matplotlib import pyplot as plt
from utils import *

########
# # Test pack / unpack_q

nx = 2
ny = 2

x_points = np.linspace(0,1,nx+1)
y_points = np.linspace(0,1,ny+1)
co_x, co_y = np.meshgrid(x_points, y_points,indexing='ij')

u_vels = np.ones((nx, ny))
v_vels = np.ones((nx,ny))
top_wall = (1,0)
bottom_wall = (0,0)
left_wall = (0,0)
right_wall = (0,0)


#### Test q packing

# for j in range(ny):
#     for i in range(nx):
#         v_vels[i,j] = i+2*j

# print(v_vels)

# q_vels = pack_q(u_vels, v_vels, nx, ny)

# print(q_vels)

# U, V = unpack_q(q_vels, nx, ny)

# print(V)
#print(V)
######

#### Test p packing
pressures = np.zeros((nx,ny))

for j in range(ny):
    for i in range(nx):
        pressures[i,j] = i+2*j
        
print(pressures)

packed = pack_p(pressures, nx, ny)

print(packed)

unpack = unpack_p(packed, nx, ny)

print(unpack)

######


#### Test Collocate Velocities
# U, V = collocate_velocity(u_vels, v_vels, nx, ny, top_wall, left_wall, right_wall, bottom_wall)

# plt.figure()
# plt.contourf(co_x, co_y, U)
# plt.title('U Velocity')
# plt.colorbar()

# plt.figure()
# plt.contourf(co_x, co_y, V)
# plt.title('V Velocity')
# plt.colorbar()

# plt.figure()
# plt.quiver(co_x, co_y, U, V)
# plt.title('Velocity Field')


# plt.show()