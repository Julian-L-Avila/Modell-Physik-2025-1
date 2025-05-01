import numpy as np

def lado_derecho(x):
    pi = np.pi
    return -pi**2 * np.cos(pi * x)

def gauss_seidel(A, b, x, iterations):
    N = len(x)
    for _ in range(iterations):
        for i in range(N):
            s = np.dot(A[i, :i], x[:i]) + np.dot(A[i, i+1:], x[i+1:])
            x[i] = (b[i] - s) / A[i, i]
    return x

xi = -1.0
xf = 2.0
vi = -1.0
vf = 1.0
nn = 11
N = nn - 2
h = (xf - xi) / (nn - 1)

A = np.zeros((N, N))
b = np.zeros(N)
x = np.zeros(N)

R = 1.0 / h**2
P = -2.0 / h**2
Q = 1.0 / h**2

A[0, 0] = P
A[1, 0] = Q
b[0] = lado_derecho(xi + h) - vi * R

for i in range(1, N - 1):
    A[i - 1, i] = R
    A[i, i] = P
    A[i + 1, i] = Q
    b[i] = lado_derecho(xi + h * (i + 1))

A[N - 2, N - 1] = R
A[N - 1, N - 1] = P
b[N - 1] = lado_derecho(xi + h * (N + 1)) - vf * Q

x = gauss_seidel(A, b, x, 1000)

with open("resultado.dat", "w") as f:
    for i in range(N):
        f.write(f"{xi + (i + 1) * h:.8f} {x[i]:.8f}\n")

