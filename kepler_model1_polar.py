import deepxde as dde
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from scipy.integrate import solve_ivp

# Imports from Cartesian common components
try:
    from common_components import (
        create_kepler_net,
        plot_loss_history,
        GM as GM_cartesian, # Use one consistent GM
        x0_val as x0_cart_ic, # Original Cartesian ICs
        y0_val as y0_cart_ic,
        vx0_val as vx0_cart_ic,
        vy0_val as vy0_cart_ic
    )
except ImportError as e:
    print(f"CRITICAL IMPORT ERROR from common_components: {e}")
    # If these critical components are missing, the script cannot run.
    exit()


# Imports from Polar common components
try:
    from common_components_polar import (
        t_begin,
        t_end,
        geom_polar,
        r0_pol,
        theta0_pol,
        drdt0_pol,
        dthetadt0_pol,
        boundary_initial_polar,
        kepler_ode_system_polar
    )
except ImportError as e:
    print(f"CRITICAL IMPORT ERROR from common_components_polar: {e}")
    # If these critical components are missing, the script cannot run.
    exit()


# Use GM from one source (already done in common_components_polar, but good for clarity)
GM = GM_cartesian

# --- Helper function to generate Cartesian solution for comparison/training data ---
def scipy_ode_system_cartesian(t, Y_cart):
    x, y, vx, vy = Y_cart[0], Y_cart[1], Y_cart[2], Y_cart[3]
    r_sq = x**2 + y**2
    r_cubed = (r_sq + 1e-9)**(1.5) # Epsilon for stability

    dvx_dt = -GM * x / r_cubed
    dvy_dt = -GM * y / r_cubed
    return [vx, vy, dvx_dt, dvy_dt]

# --- Generate Training Data (r, theta from Cartesian solution) ---
Y0_cartesian = [x0_cart_ic, y0_cart_ic, vx0_cart_ic, vy0_cart_ic]
num_data_points = 20
t_eval_data = np.linspace(t_begin, t_end, num_data_points)

# Use higher precision for solve_ivp if generating reference data
sol_cartesian_data = solve_ivp(
    scipy_ode_system_cartesian,
    (t_begin, t_end),
    Y0_cartesian,
    dense_output=True,
    t_eval=t_eval_data,
    rtol=1e-8,  # Reduced error tolerance
    atol=1e-8   # Reduced error tolerance
)

observe_t_data = sol_cartesian_data.t.reshape(-1, 1)
x_data_cart = sol_cartesian_data.y[0, :]
y_data_cart = sol_cartesian_data.y[1, :]

r_data_polar = np.sqrt(x_data_cart**2 + y_data_cart**2)
# np.unwrap is important for theta to avoid jumps, making it easier for the NN
theta_data_polar = np.unwrap(np.arctan2(y_data_cart, x_data_cart))

observe_rtheta_data = np.vstack((r_data_polar, theta_data_polar)).T

# --- DeepXDE ICs and Data BCs for Polar Coordinates ---
# component=0 for r, 1 for theta, 2 for dr/dt, 3 for dtheta/dt
ic_r = dde.icbc.IC(geom_polar, lambda t: r0_pol, boundary_initial_polar, component=0)
ic_theta = dde.icbc.IC(geom_polar, lambda t: theta0_pol, boundary_initial_polar, component=1)
ic_drdt = dde.icbc.IC(geom_polar, lambda t: drdt0_pol, boundary_initial_polar, component=2)
ic_dthetadt = dde.icbc.IC(geom_polar, lambda t: dthetadt0_pol, boundary_initial_polar, component=3)

# Data-driven boundary conditions for r(t) and theta(t)
bc_data_r = dde.icbc.PointSetBC(observe_t_data, observe_rtheta_data[:, 0:1], component=0)
bc_data_theta = dde.icbc.PointSetBC(observe_t_data, observe_rtheta_data[:, 1:2], component=1)

bcs_polar = [ic_r, ic_theta, ic_drdt, ic_dthetadt, bc_data_r, bc_data_theta]

# --- TimePDE Data Object ---
data_polar = dde.data.TimePDE(
    geom_polar,
    kepler_ode_system_polar, # The ODE system defined in common_components_polar
    bcs_polar,
    num_domain=500,
    num_boundary=20, # Increased num_boundary slightly for better IC enforcement sampling
    num_initial=0,   # ICs handled by num_boundary via boundary_initial_polar
    train_distribution="pseudo"
)

# --- Network, Model Compilation, and Training ---
model_name = "Model1_Polar"
output_dir_model1_polar = f"{model_name}_outputs"
os.makedirs(output_dir_model1_polar, exist_ok=True)

# Network: 1 input (t), 4 outputs (r, theta, dr/dt, dtheta/dt)
# Using 64 neurons as specified in the plan to test flexibility
net_polar = create_kepler_net(
    input_dims=1,
    output_dims=4,
    num_hidden_layers=3,
    num_neurons_per_layer=64, # Changed to 64
    hidden_activation='tanh'
)

model_polar = dde.Model(data_polar, net_polar)

# Loss weights: 4 PDE residuals, 4 ICs, 2 Data BCs = 10 components
loss_weights_polar = [1, 1, 1, 1,  100, 100, 100, 100,  10, 10]
model_polar.compile("adam", lr=1e-3, loss_weights=loss_weights_polar)

iterations = 1000 # Reduced for subtask verification
display_every = 200
print(f"Starting training for {model_name}...")
losshistory, train_state = model_polar.train(iterations=iterations, display_every=display_every)
print(f"Training finished for {model_name}.")

# --- Plotting and Saving Results ---
plot_loss_history(losshistory, model_name, output_dir=output_dir_model1_polar)
# Note: dde.saveplot's isplot=True might try to open interactive window. Set to False.
dde.saveplot(losshistory, train_state, issave=True, isplot=False, output_dir=output_dir_model1_polar)

t_plot = np.linspace(t_begin, t_end, 200)
y_pred_polar_plot = model_polar.predict(t_plot.reshape(-1, 1))
r_pred = y_pred_polar_plot[:, 0]
theta_pred = y_pred_polar_plot[:, 1]
# drdt_pred = y_pred_polar_plot[:, 2] # Not plotted directly, but available
# dthetadt_pred = y_pred_polar_plot[:, 3] # Not plotted directly

# Get exact Cartesian solution for comparison plot
sol_cartesian_exact_plot = solve_ivp(
    scipy_ode_system_cartesian,
    (t_begin, t_end),
    Y0_cartesian,
    dense_output=True,
    t_eval=t_plot,
    rtol=1e-8,
    atol=1e-8
)
x_exact_cart_plot = sol_cartesian_exact_plot.y[0, :]
y_exact_cart_plot = sol_cartesian_exact_plot.y[1, :]
# Convert exact Cartesian to Polar for r(t), theta(t) comparison
r_exact_polar_plot = np.sqrt(x_exact_cart_plot**2 + y_exact_cart_plot**2)
theta_exact_polar_plot = np.unwrap(np.arctan2(y_exact_cart_plot, x_exact_cart_plot))


# Plot r(t) and theta(t)
plt.figure(figsize=(12, 6)) # Adjusted figure size slightly
plt.subplot(1, 2, 1)
plt.plot(t_plot, r_pred, 'r--', label='r_pred (PINN)')
plt.plot(t_plot, r_exact_polar_plot, 'b-', label='r_exact (Numerical)')
plt.scatter(observe_t_data.flatten(), r_data_polar, c='k', marker='x', s=50, label='r_data (Training)')
plt.xlabel('Time (t)')
plt.ylabel('r(t)')
plt.legend()
plt.title(f'{model_name} - r(t) Comparison')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(t_plot, theta_pred, 'r--', label='theta_pred (PINN)')
plt.plot(t_plot, theta_exact_polar_plot, 'b-', label='theta_exact (Numerical)')
plt.scatter(observe_t_data.flatten(), theta_data_polar, c='k', marker='x', s=50, label='theta_data (Training)')
plt.xlabel('Time (t)')
plt.ylabel('$\\theta(t)$ (rad)') # Used LaTeX for theta
plt.legend()
plt.title(f'{model_name} - $\\theta(t)$ Comparison') # Used LaTeX for theta
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir_model1_polar, f"{model_name}_polar_vars_time_comparison.png"))
plt.close()

# Plot trajectory (convert PINN polar to Cartesian)
x_pred_cart = r_pred * np.cos(theta_pred)
y_pred_cart = r_pred * np.sin(theta_pred)

plt.figure(figsize=(8, 8))
plt.plot(x_pred_cart, y_pred_cart, 'r--', label='Trajectory_pred (PINN Polar)')
plt.plot(x_exact_cart_plot, y_exact_cart_plot, 'b-', label='Trajectory_exact (Numerical Cartesian)')
# For plotting training data, use the original Cartesian points used to derive r_data, theta_data
plt.scatter(x_data_cart, y_data_cart, c='k', marker='x', s=50, label='Training Data (Cartesian)')
plt.xlabel('x')
plt.ylabel('y')
plt.axis('equal')
plt.legend()
plt.title(f'{model_name} - Trajectory Comparison (x-y)')
plt.grid(True)
plt.savefig(os.path.join(output_dir_model1_polar, f"{model_name}_trajectory_comparison_xy.png"))
plt.close()

print(f"Finished training and plotting for {model_name}. Outputs in {output_dir_model1_polar}/")
print(f"To see plots, check the directory: {output_dir_model1_polar}")

if __name__ == "__main__":
    # This structure ensures the script runs top-to-bottom when executed.
    print(f"Executing {model_name} script...")
    pass
