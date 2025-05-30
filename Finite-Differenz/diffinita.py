import numpy as np
import sys

def ladoDerecho(x):
    """
    Calculates the right-hand side of the differential equation.

    Args:
        x (float): The value of x.

    Returns:
        float: The result of the calculation.
    """
    pi = np.pi
    return -pi * pi * np.cos(pi * x)

def gaussSeidel(A, x, b, nx, iter_max=10000):
    """
    Solves a system of linear equations using the Gauss-Seidel method.

    Args:
        A (numpy.ndarray): The coefficient matrix.
        x (numpy.ndarray): The initial guess for the solution vector (modified in place).
        b (numpy.ndarray): The right-hand side vector.
        nx (int): The dimension of the system.
        iter_max (int, optional): The maximum number of iterations. Defaults to 10000.
    """
    for m in range(iter_max):
        for i in range(nx):
            sum_val = 0.0
            for j in range(nx):
                if i != j:
                    sum_val += A[i, j] * x[j]  # Corrected indexing to A[i, j]
            x[i] = (b[i] - sum_val) / A[i, i]

def fdm1d():
    """
    Solves a 1D finite difference problem.
    """
    # Leer argumento N (nodos interiores)
    if len(sys.argv) < 2:
        print("Error: debe pasar N (nodos interiores) como argumento.")
        sys.exit(1)
    try:
        N = int(sys.argv[1])
    except ValueError:
        print("Error: N debe ser entero positivo.")
        sys.exit(1)

    if N < 1:
        print("Error: N debe ser entero positivo.")
        sys.exit(1)

    nn = N + 2  # incluir puntos de frontera

    # Parámetros del problema
    xi = -1.0
    xf = 2.0
    vi = -1.0
    vf = 1.0
    h = (xf - xi) / (nn - 1)

    # Reservar memoria
    A = np.zeros((N, N))
    b = np.zeros(N)
    x = np.zeros(N)

    R = 1.0 / (h * h)
    P = -2.0 / (h * h)
    Q = 1.0 / (h * h)

    # Primer renglón
    A[0, 0] = P
    if N > 1:
        A[0, 1] = Q
    y = ladoDerecho(xi + h)
    b[0] = y - vi * R

    # Renglones intermedios
    for i in range(1, N - 1):
        A[i, i - 1] = R
        A[i, i] = P
        A[i, i + 1] = Q
        y = ladoDerecho(xi + (i + 1) * h) # changed i to i+1 to match fortran
        b[i] = y

    # Último renglón
    if N > 1:
        A[N - 1, N - 2] = R
    A[N - 1, N - 1] = P
    y = ladoDerecho(xi + N * h)
    b[N - 1] = y - vf * Q

    # Resolver sistema por Gauss-Seidel
    gaussSeidel(A, x, b, N, 10000)

    # Guardar resultados
    filename = f"resultado_{N}.dat" # changed formatting to use f-strings
    try:
        with open(filename, "w") as f:
            f.write(f"{xi:10.5f} {vi:10.5f}\n")
            for i in range(N):
                f.write(f"{xi + (i+1) * h:10.5f} {x[i]:10.5f}\n") # changed i to i+1 to match fortran
            f.write(f"{xf:10.5f} {vf:10.5f}\n")
    except Exception as e:
        print(f"An error occurred while saving the results: {e}")

if __name__ == "__main__":
    fdm1d()
