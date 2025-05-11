import numpy as np
import time

def initialize_grid_and_boundaries(Nx, Ny, h, xmin, ymin, Lx, Ly):
    """
    Initializes the potential phi, source term rho, and boundary mask.
    Sets Dirichlet boundary conditions and calculates rho(x,y) = x/y + y/x.
    """
    # Grid points (including boundaries): Nx+1 points in x, Ny+1 points in y
    phi = np.zeros((Nx + 1, Ny + 1), dtype=np.float64)
    rho = np.zeros((Nx + 1, Ny + 1), dtype=np.float64)
    is_boundary = np.zeros((Nx + 1, Ny + 1), dtype=bool)

    # Physical coordinates for each grid point
    # x_coords[i] = xmin + i * h
    # y_coords[j] = ymin + j * h
    # Note: For rho calculation, we'll compute x, y on the fly or use meshgrid

    # Mark boundary points
    is_boundary[0, :] = True   # Left boundary (x = xmin)
    is_boundary[Nx, :] = True  # Right boundary (x = xmin + Lx)
    is_boundary[:, 0] = True   # Bottom boundary (y = ymin)
    is_boundary[:, Ny] = True  # Top boundary (y = ymin + Ly)

    # Apply Dirichlet boundary conditions
    for i in range(Nx + 1):
        x = xmin + i * h
        if x <= 0: # Should not happen with xmin=1
            phi[i, 0] = 0.0
        else:
            phi[i, 0] = 1.0

    for j in range(Ny + 1):
        y = ymin + j * h
        if y <= 0: # Should not happen with ymin=1
            phi[0, j] = 0.0
        else:
            phi[0, j] = 1.0

    for i in range(Nx + 1):
        x = xmin + i * h
        if x <= 0 or (4 * x) <= 0: # Should not happen with xmin=1
            phi[i, Ny] = 0.0
        else:
            phi[i, Ny] = np.exp(x)

    for j in range(Ny + 1):
        y = ymin + j * h
        if y <= 0 or (2 * y) <= 0: # Should not happen with ymin=1
            phi[Nx, j] = 0.0
        else:
            phi[Nx, j] = np.exp(2 * y)
            
    # For boundary points, rho is not used by the solver directly for updating phi,
    # but we can define it. The Fortran code effectively set rho=0 on boundaries
    # within the initialization loop. Here we will set it for interior points
    # and it will remain 0 for boundaries unless explicitly set.

    # Correctly calculate rho for interior points, it will be 0 on boundaries
    # as per rho array initialization.
    for i in range(1, Nx): # Interior points in x
        x = xmin + i * h
        for j in range(1, Ny): # Interior points in y
            y = ymin + j * h
            # With xmin=1, ymin=1, x and y will be > 0, so no division by zero.
            if abs(y) < 1e-12 or abs(x) < 1e-12 : # Safety for general case
                 # This case should not be hit for the current domain [1,2]x[1,2]
                rho[i,j] = 1.0e20 # Or handle as an error
                print(f"Warning: x or y is near zero for rho calc at interior: ({x},{y})")
            else:
                rho[i, j] = (x**2 + y**2) * np.exp(x * y)
                
    return phi, rho, is_boundary

def poisson_solver(Nx, Ny, h, eps, max_iter, phi, rho, is_boundary):
    """
    Solves the Poisson equation using Gauss-Seidel iteration.
    phi is modified in-place.
    """
    iter_count = 0
    # omega = 1.8 # Optional: SOR parameter (1.0 for Gauss-Seidel)

    while True:
        max_error = 0.0
        
        # Iterate over interior grid points
        # The order of loops (i then j, or j then i) can slightly affect
        # convergence path but generally not the final result for GS.
        # Fortran code did j (columns), then i (rows).
        for j in range(1, Ny): # Corresponds to Fortran j=1,Ny-1
            for i in range(1, Nx): # Corresponds to Fortran i=1,Nx-1
                if not is_boundary[i, j]: # Should always be true for this loop range
                    phi_old_ij = phi[i, j]
                    
                    phi_new_ij = 0.25 * (phi[i + 1, j] + phi[i - 1, j] +
                                         phi[i, j + 1] + phi[i, j - 1] -
                                         h**2 * rho[i, j])
                    
                    # Optional: Successive Over-Relaxation (SOR)
                    # phi[i, j] = (1.0 - omega) * phi_old_ij + omega * phi_new_ij
                    phi[i, j] = phi_new_ij # Standard Gauss-Seidel update
                                        
                    dphi = abs(phi[i, j] - phi_old_ij) # Error uses new vs old of current iter
                    if dphi > max_error:
                        max_error = dphi
        
        iter_count += 1
        
        if iter_count % 500 == 0: # Optional: print progress
             print(f"Iteration: {iter_count}, Max Error: {max_error:.5e}")

        if max_error < eps:
            print(f"Converged after {iter_count} iterations.")
            break
        
        if iter_count >= max_iter:
            print(f"Warning: Maximum number of iterations ({max_iter}) reached.")
            print(f"Final maximum error = {max_error:.5e}")
            break
            
    print(f"Iterations to converge: {iter_count}, Final maximum error = {max_error:.5e}")
    return phi # phi has been modified in-place, but returning it is Pythonic

def save_results(Nx, Ny, h, xmin, ymin, phi, filename="data_poisson-01-python.txt"):
    """
    Saves the potential phi to a file.
    The output format is similar to the Fortran version for gnuplot.
    """
    with open(filename, "w") as f:
        f.write("# x y Potential(numerical)\n")
        for j in range(Ny + 1): # y varies slower for gnuplot pm3d
            for i in range(Nx + 1):
                x = xmin + i * h
                y = ymin + j * h
                f.write(f"{x:12.6f} {y:12.6f} {phi[i, j]:17.9f}\n")
            f.write("\n") # Blank line after each row of x values for gnuplot splot
    print(f"Results saved to {filename}")

# --- Main script execution ---
if __name__ == "__main__":
    # Fixed Parameters
    Nx = 100  # Number of intervals in X
    Ny = 50  # Number of intervals in Y (for Lx=1, Ly=1 => Nx=Ny for hx=hy)
    eps = 1.0e-7  # Convergence criterion

    # Domain definition
    xmin = 0.0
    ymin = 0.0
    Lx = 2.0  # Length of domain in X (xmax = xmin + Lx = 2.0)
    Ly = 1.0  # Length of domain in Y (ymax = ymin + Ly = 2.0)

    max_iter = 100000 # Maximum iterations for the solver

    # Calculate step size h (assuming square grid hx=hy)
    if abs(Lx / Nx - Ly / Ny) > 1e-9:
        print("Error: Grid is not square with the given Nx, Ny, Lx, Ly.")
        print(f"Lx/Nx = {Lx/Nx}, Ly/Ny = {Ly/Ny}")
        print(f"For a square grid with Lx={Lx}, Ly={Ly}, Nx={Nx}, Ny should be {int(Ly/Lx * Nx)}")
        exit()
    h = Lx / Nx

    start_time = time.time()

    # Initialize grid, potential, source term, and boundary conditions
    phi, rho, is_boundary = initialize_grid_and_boundaries(Nx, Ny, h, xmin, ymin, Lx, Ly)
    
    # Solve the Poisson equation
    phi_solution = poisson_solver(Nx, Ny, h, eps, max_iter, phi, rho, is_boundary)
    
    # Save results
    save_results(Nx, Ny, h, xmin, ymin, phi_solution)

    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.3f} seconds")
