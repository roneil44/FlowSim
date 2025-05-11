########
# Used to test conjugant gradient solver
# Riley O'Neil
#
########


from conjugant_solver import *

#Initializations
x0 = [2,1]
b = [3,2]
A = [[5,1],[1,8]]


# Test Ax
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