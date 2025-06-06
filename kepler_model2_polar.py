import deepxde as dde
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from scipy.integrate import solve_ivp

# Imports from Cartesian common components for network creation, plotting, and original ICs/GM
try:
    from common_components import (
        create_kepler_net,
        plot_loss_history,
        GM as GM_cartesian,
        x0_val as x0_cart_ic,
        y0_val as y0_cart_ic,
        vx0_val as vx0_cart_ic,
        vy0_val as vy0_cart_ic
    )
except ImportError as e:
    print(f"CRITICAL IMPORT ERROR from common_components: {e}")
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
        kepler_ode_system_polar # This is the key physics
    )
except ImportError as e:
    print(f"CRITICAL IMPORT ERROR from common_components_polar: {e}")
    exit()

GM = GM_cartesian # Consistent GM

# --- Helper function for Cartesian solution (for comparison only) ---
def scipy_ode_system_cartesian(t, Y_cart):
    x, y, vx, vy = Y_cart[0], Y_cart[1], Y_cart[2], Y_cart[3]
    r_sq = x**2 + y**2
    r_cubed = (r_sq + 1e-9)**(1.5) # Epsilon for stability
    dvx_dt = -GM * x / r_cubed
    dvy_dt = -GM * y / r_cubed
    return [vx, vy, dvx_dt, dvy_dt]

# --- DeepXDE ICs for Polar Coordinates ---
ic_r = dde.icbc.IC(geom_polar, lambda t: r0_pol, boundary_initial_polar, component=0)
ic_theta = dde.icbc.IC(geom_polar, lambda t: theta0_pol, boundary_initial_polar, component=1)
ic_drdt = dde.icbc.IC(geom_polar, lambda t: drdt0_pol, boundary_initial_polar, component=2)
ic_dthetadt = dde.icbc.IC(geom_polar, lambda t: dthetadt0_pol, boundary_initial_polar, component=3)

# Model 2 uses only ICs, no data BCs
bcs_polar_m2 = [ic_r, ic_theta, ic_drdt, ic_dthetadt]

# --- TimePDE Data Object ---
data_polar_m2 = dde.data.TimePDE(
    geom_polar,
    kepler_ode_system_polar,
    bcs_polar_m2,
    num_domain=500,
    num_boundary=20, # Increased num_boundary from plan's 4 to 20 for better IC sampling
    num_initial=0,   # ICs handled by num_boundary via boundary_initial_polar
    train_distribution="pseudo" # Added train_distribution
)

# --- Network, Model Compilation, and Training ---
model_name = "Model2_Polar"
output_dir_model2_polar = f"{model_name}_outputs"
os.makedirs(output_dir_model2_polar, exist_ok=True)

net_polar_m2 = create_kepler_net(
    input_dims=1,
    output_dims=4,
    num_hidden_layers=3,
    num_neurons_per_layer=64, # Consistent with Model1_Polar
    hidden_activation='tanh'
)

model_polar_m2 = dde.Model(data_polar_m2, net_polar_m2)

# Loss weights: 4 PDE res, 4 ICs
loss_weights_polar_m2 = [1, 1, 1, 1,  100, 100, 100, 100]
model_polar_m2.compile("adam", lr=1e-3, loss_weights=loss_weights_polar_m2)

iterations = 1000 # Reduced for subtask; use more for actual run (e.g., 40000)
display_every=200
print(f"Starting training for {model_name} (PDEs and ICs only)...")
losshistory, train_state = model_polar_m2.train(iterations=iterations, display_every=display_every)
print(f"Training finished for {model_name}.")

# --- Plotting and Saving Results ---
plot_loss_history(losshistory, model_name, output_dir=output_dir_model2_polar)
# Note: dde.saveplot's isplot=True might try to open interactive window. Set to False.
dde.saveplot(losshistory, train_state, issave=True, isplot=False, output_dir=output_dir_model2_polar)

t_plot = np.linspace(t_begin, t_end, 200)
y_pred_polar_plot_m2 = model_polar_m2.predict(t_plot.reshape(-1, 1))
r_pred_m2 = y_pred_polar_plot_m2[:, 0]
theta_pred_m2 = y_pred_polar_plot_m2[:, 1]

# Get exact Cartesian solution for comparison plot
Y0_cartesian_ic = [x0_cart_ic, y0_cart_ic, vx0_cart_ic, vy0_cart_ic]
sol_cartesian_exact_plot_m2 = solve_ivp(
    scipy_ode_system_cartesian,
    (t_begin, t_end),
    Y0_cartesian_ic,
    dense_output=True,
    t_eval=t_plot,
    rtol=1e-8,
    atol=1e-8
)
x_exact_cart_plot_m2 = sol_cartesian_exact_plot_m2.y[0, :]
y_exact_cart_plot_m2 = sol_cartesian_exact_plot_m2.y[1, :]
# Convert exact Cartesian to Polar for r(t), theta(t) comparison
r_exact_polar_plot_m2 = np.sqrt(x_exact_cart_plot_m2**2 + y_exact_cart_plot_m2**2)
theta_exact_polar_plot_m2 = np.unwrap(np.arctan2(y_exact_cart_plot_m2, x_exact_cart_plot_m2))

# Plot r(t) and theta(t)
plt.figure(figsize=(12, 6)) # Adjusted figure size
plt.subplot(1, 2, 1)
plt.plot(t_plot, r_pred_m2, 'r--', label='r_pred (PINN)')
plt.plot(t_plot, r_exact_polar_plot_m2, 'b-', label='r_exact (Numerical)')
plt.xlabel('Time (t)')
plt.ylabel('r(t)')
plt.legend()
plt.title(f'{model_name} - r(t) Comparison')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(t_plot, theta_pred_m2, 'r--', label='$\\theta_{pred}$ (PINN)') # Using LaTeX
plt.plot(t_plot, theta_exact_polar_plot_m2, 'b-', label='$\\theta_{exact}$ (Numerical)') # Using LaTeX
plt.xlabel('Time (t)')
plt.ylabel('$\\theta(t)$ (rad)')
plt.legend()
plt.title(f'{model_name} - $\\theta(t)$ Comparison')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir_model2_polar, f"{model_name}_polar_vars_time_comparison.png"))
plt.close()

# Plot trajectory (convert PINN polar to Cartesian)
x_pred_cart_m2 = r_pred_m2 * np.cos(theta_pred_m2)
y_pred_cart_m2 = r_pred_m2 * np.sin(theta_pred_m2)

plt.figure(figsize=(8, 8))
plt.plot(x_pred_cart_m2, y_pred_cart_m2, 'r--', label='Trajectory_pred (PINN Polar)')
plt.plot(x_exact_cart_plot_m2, y_exact_cart_plot_m2, 'b-', label='Trajectory_exact (Numerical Cartesian)')
plt.xlabel('x')
plt.ylabel('y')
plt.axis('equal')
plt.legend()
plt.title(f'{model_name} - Trajectory Comparison (x-y)')
plt.grid(True)
plt.savefig(os.path.join(output_dir_model2_polar, f"{model_name}_trajectory_comparison_xy.png"))
plt.close()

print(f"Finished training and plotting for {model_name}. Outputs in {output_dir_model2_polar}/")
print(f"To see plots, check the directory: {output_dir_model2_polar}")

if __name__ == "__main__":
    print(f"Executing {model_name} script...")
    pass
