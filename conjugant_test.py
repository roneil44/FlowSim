########
# Used to test conjugant gradient solver
# Riley O'Neil
#
########


from conjugant_solver import *

#Initializations
x0 = [2,1]
b = [3,2]
A = [[1,8],[5,1]]


# Test Ax
print(Ax(A, x0))