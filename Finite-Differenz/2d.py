import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def right_hand_side_2d(x, y):
    """
    Calculates the right-hand side of the 2D differential equation.

    Args:
        x (float): The x-coordinate.
        y (float): The y-coordinate.

    Returns:
        float: The value of the function f(x, y).
    """
    pi = np.pi
    return -pi**2 * np.cos(pi * x * y)

def gauss_seidel_2d(u, f, hx, hy, nx, ny, iter_max=10000, tolerance=1e-6):
    """
    Solves a 2D system using the Gauss-Seidel method on a grid.

    Args:
        u (numpy.ndarray): The initial guess for the solution grid (modified in place).
                         Includes boundary points.
        f (numpy.ndarray): The right-hand side grid values.
        hx (float): The grid spacing in the x-direction.
        hy (float): The grid spacing in the y-direction.
        nx (int): The number of interior points in the x-direction.
        ny (int): The number of interior points in the y-direction.
        iter_max (int, optional): The maximum number of iterations. Defaults to 10000.
        tolerance (float, optional): The convergence tolerance. Defaults to 1e-6.
    """
    # Check if grid spacing is the same (for simplified update formula)
    if abs(hx - hy) > 1e-9:
         print("Warning: hx and hy are not equal. Using a simplified update assuming hx=hy.")

    h_sq = hx * hy # Using h*h from the simplified formula if hx=hy

    print("Starting Gauss-Seidel iterations...")
    for iteration in range(iter_max):
        max_diff = 0.0
        # Iterate over interior points (1 to nx, 1 to ny)
        # Note: In numpy arrays, indices are 0-based.
        # The boundary points are at indices 0 and nx+1 (x-dir), 0 and ny+1 (y-dir).
        # Interior points are from 1 to nx (inclusive) and 1 to ny (inclusive).
        for i in range(1, nx + 1):
            for j in range(1, ny + 1):
                old_u_ij = u[i, j]

                # Gauss-Seidel update using the discretized equation
                # u_ij = 1/4 * (u_{i-1,j} + u_{i+1,j} + u_{i,j-1} + u_{i,j+1} - h^2 * f_{i,j})
                u[i, j] = 0.25 * (u[i-1, j] + u[i+1, j] + u[i, j-1] + u[i, j+1] - h_sq * f[i, j])

                diff = abs(u[i, j] - old_u_ij)
                if diff > max_diff:
                    max_diff = diff

        # Optional: Print progress
        if (iteration + 1) % 100 == 0 or iteration == 0:
             print(f"Iteration {iteration + 1}/{iter_max}, Max change: {max_diff:.2e}")

        # Check for convergence
        if max_diff < tolerance:
            print(f"Converged after {iteration + 1} iterations. Max change: {max_diff:.2e}")
            break

    if iteration == iter_max - 1 and max_diff >= tolerance:
        print(f"Gauss-Seidel reached maximum iterations ({iter_max}) without converging to tolerance {tolerance}. Final max change: {max_diff:.2e}")


def fdm2d_solve_and_plot():
    """
    Solves a 2D finite difference problem and saves plots of the solution.
    """
    # Domain and grid parameters
    xmin, xmax = -1.0, 2.0
    ymin, ymax = -1.0, 2.0
    nx = 50 # Number of interior points in x-direction
    ny = 50 # Number of interior points in y-direction

    # Total number of points including boundaries
    total_nx = nx + 2
    total_ny = ny + 2

    # Grid spacing
    hx = (xmax - xmin) / (total_nx - 1)
    hy = (ymax - ymin) / (total_ny - 1)

    # Create grid points
    x_points = np.linspace(xmin, xmax, total_nx)
    y_points = np.linspace(ymin, ymax, total_ny)
    X, Y = np.meshgrid(x_points, y_points) # For plotting later

    # Initialize the solution grid u (including boundary points)
    # Set initial guess (e.g., zeros) and apply boundary conditions
    u = np.zeros((total_nx, total_ny))

    # Apply Dirichlet Boundary Conditions (u = 0 on all boundaries)
    # The grid is already initialized with zeros, so this is implicitly done.
    # If boundary conditions were non-zero, we would set the edges here:
    # u[0, :] = ...
    # u[total_nx-1, :] = ...
    # u[:, 0] = ...
    # u[:, total_ny-1] = ...

    # Create the right-hand side grid f
    # Note: f needs to be evaluated at the *interior* grid points (x_i, y_j)
    # We store f on a grid of the same size as u for convenience, but only use
    # the values corresponding to interior points during the solve.
    f_grid = np.zeros((total_nx, total_ny))
    for i in range(1, nx + 1):
        for j in range(1, ny + 1):
            f_grid[i, j] = right_hand_side_2d(x_points[i], y_points[j])


    # Solve the system using Gauss-Seidel
    print(f"Solving 2D PDE on a {nx}x{ny} interior grid (total {total_nx}x{total_ny})...")
    gauss_seidel_2d(u, f_grid, hx, hy, nx, ny, iter_max=20000, tolerance=1e-7) # Increased iter_max and tolerance slightly
    print("Solver finished.")

    # --- Plotting and Saving ---

    # 1. General 3D View
    print("Saving general 3D view...")
    fig1 = plt.figure(figsize=(10, 7))
    ax1 = fig1.add_subplot(111, projection='3d')
    surf1 = ax1.plot_surface(X, Y, u, cmap='viridis', linewidth=0, antialiased=False)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('u(x, y)')
    ax1.set_title(f'Solution of $u_{{xx}} + u_{{yy}} = -\\pi^2 \cos(\\pi xy)$ ({nx}x{ny} grid)')
    fig1.colorbar(surf1, shrink=0.5, aspect=5)
    plt.savefig('solution_3d_general.jpeg', dpi=300)
    plt.close(fig1) # Close figure to free memory
    print("Saved: solution_3d_general.jpeg")

    # 2. XY Projection (Contour Plot)
    print("Saving XY projection (contour plot)...")
    fig2 = plt.figure(figsize=(8, 6))
    ax2 = fig2.add_subplot(111)
    contour_levels = np.linspace(u.min(), u.max(), 20) # Or choose specific levels
    cp = ax2.contourf(X, Y, u, levels=contour_levels, cmap='viridis')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title(f'Solution Contour Plot (XY Projection, {nx}x{ny} grid)')
    fig2.colorbar(cp, label='u(x, y)')
    ax2.set_aspect('equal', adjustable='box') # Make sure aspect ratio is equal
    plt.savefig('solution_xy_projection.jpeg', dpi=300)
    plt.close(fig2)
    print("Saved: solution_xy_projection.jpeg")


    # 3. ZY Projection (Rotated 3D View)
    print("Saving ZY projection (rotated 3D view)...")
    fig3 = plt.figure(figsize=(10, 7))
    ax3 = fig3.add_subplot(111, projection='3d')
    surf3 = ax3.plot_surface(X, Y, u, cmap='viridis', linewidth=0, antialiased=False)
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('u(x, y)')
    ax3.set_title(f'Solution ZY View ({nx}x{ny} grid)')
    fig3.colorbar(surf3, shrink=0.5, aspect=5)
    # Rotate view to look approximately along the positive X axis
    ax3.view_init(elev=15, azim=90) # elev controls vertical angle, azim horizontal
    plt.savefig('solution_zy_projection.jpeg', dpi=300)
    plt.close(fig3)
    print("Saved: solution_zy_projection.jpeg")

    print("All plots saved.")


if __name__ == "__main__":
    fdm2d_solve_and_plot()
