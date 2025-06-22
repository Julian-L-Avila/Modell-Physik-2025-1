import deepxde as dde
import numpy as np
import tensorflow as tf
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import os

# Import from common_components
try:
    from common_components import GM, t_begin, t_end, geom, \
                                x0_val, y0_val, vx0_val, vy0_val, boundary_initial, \
                                create_kepler_net, plot_loss_history # Added generalized net and plot_loss_history
except ImportError:
    print("Error: common_components.py not found or an import failed.")
    print("Ensure common_components.py is in the same directory or accessible in PYTHONPATH, and contains all required functions.")
    exit()

# 1. Define the ODE System for DeepXDE
def kepler_ode_system(t_in, y_vec):
    """
    Defines the system of ODEs for the Kepler problem for DeepXDE.
    y_vec corresponds to [x, y, vx, vy].
    t_in is the time coordinate.
    """
    x_pos, y_pos, vx_vel, vy_vel = y_vec[:, 0:1], y_vec[:, 1:2], y_vec[:, 2:3], y_vec[:, 3:4]

    # Calculate derivatives using dde.grad.jacobian
    # For dy/dt, y is y_vec, x is t_in.
    # We need the derivative of each component of y_vec w.r.t. t_in

    # jacobian computes d(y_vec_i)/d(t_in_j)
    # Here y_vec has 4 components (x,y,vx,vy) and t_in has 1 component (t)
    # So, dy_dt will be a tensor where dy_dt[k, i] is d(y_vec[k,i])/d(t_in[k,0])
    # We want d(y_vec[:,i])/d(t_in) essentially.
    # dde.grad.jacobian(y_vec, t_in, i=0, j=0) gives d(y_vec[:,0])/d(t_in[:,0])

    dx_dt = dde.grad.jacobian(y_vec, t_in, i=0)
    dy_dt = dde.grad.jacobian(y_vec, t_in, i=1)
    dvx_dt = dde.grad.jacobian(y_vec, t_in, i=2)
    dvy_dt = dde.grad.jacobian(y_vec, t_in, i=3)

    r_squared = x_pos**2 + y_pos**2
    # Add epsilon for numerical stability, especially for r=0 or during early training
    epsilon_tf = tf.constant(1e-8, dtype=y_vec.dtype) # Ensure dtype matches y_vec

    r = tf.sqrt(tf.maximum(r_squared, epsilon_tf)) # Avoid sqrt(0) or negative if r_squared is momentarily negative
    r_cubed = tf.maximum(r**3, epsilon_tf) # Avoid division by zero

    # Define the residuals of the ODEs
    res1 = dx_dt - vx_vel
    res2 = dy_dt - vy_vel
    res3 = dvx_dt + GM * x_pos / r_cubed
    res4 = dvy_dt + GM * y_pos / r_cubed

    return [res1, res2, res3, res4]

# 2. Generate Training Data (Numerical Solution using SciPy)
def scipy_ode_system(t_sci, Y_sci):
    """
    Defines the system of ODEs for SciPy's solve_ivp.
    Y_sci = [x, y, vx, vy]
    """
    x_pos, y_pos, vx_vel, vy_vel = Y_sci[0], Y_sci[1], Y_sci[2], Y_sci[3]

    r_squared = x_pos**2 + y_pos**2
    epsilon_np = 1e-8 # Prevent division by zero for numpy

    # Ensure r_squared is not zero before sqrt, or handle r_cubed directly
    if r_squared < epsilon_np: # Effectively if r is very close to 0
        r_cubed = epsilon_np
    else:
        r_cubed = r_squared * np.sqrt(r_squared)
        r_cubed = np.maximum(r_cubed, epsilon_np) # Final check

    dvx_dt_sci = -GM * x_pos / r_cubed
    dvy_dt_sci = -GM * y_pos / r_cubed
    return [vx_vel, vy_vel, dvx_dt_sci, dvy_dt_sci]

Y0 = [x0_val, y0_val, vx0_val, vy0_val]
num_data_points = 20
t_eval_data = np.linspace(t_begin, t_end, num_data_points)

sol_data = solve_ivp(scipy_ode_system, (t_begin, t_end), Y0, dense_output=True, t_eval=t_eval_data, method='RK45')
observe_t_data = sol_data.t.reshape(-1, 1)
observe_xy_data = sol_data.y[:2, :].T  # Shape: (num_data_points, 2) for [x, y]

# 3. Define DeepXDE ICs and Data BCs
ic_x = dde.icbc.IC(geom, lambda t: x0_val, boundary_initial, component=0)
ic_y = dde.icbc.IC(geom, lambda t: y0_val, boundary_initial, component=1)
ic_vx = dde.icbc.IC(geom, lambda t: vx0_val, boundary_initial, component=2)
ic_vy = dde.icbc.IC(geom, lambda t: vy0_val, boundary_initial, component=3)

# Using observed data as PointSetBC
# component=0 means we are providing data for the 0-th output of the network (x)
# component=1 means we are providing data for the 1-st output of the network (y)
bc_data_x = dde.icbc.PointSetBC(observe_t_data, observe_xy_data[:, 0:1], component=0)
bc_data_y = dde.icbc.PointSetBC(observe_t_data, observe_xy_data[:, 1:2], component=1)

bcs = [ic_x, ic_y, ic_vx, ic_vy, bc_data_x, bc_data_y]

# 4. Create TimePDE Data Object
# num_domain: points for ODE residuals.
# num_boundary & num_initial: points for initial/boundary conditions.
# For ICs, only t_begin is relevant. DeepXDE handles this.
# PointSetBC uses its own points.
data = dde.data.TimePDE(
    geom,
    kepler_ode_system,
    bcs,
    num_domain=500,       # Collocation points for ODE residuals
    num_boundary=20,      # Points on the boundary of the time domain (t_begin and t_end)
    num_initial=0,        # ICs are handled by BC objects and their boundary functions
    train_distribution="pseudo" # Using pseudo-random for better coverage
)

# 5. Network, Model Compilation, and Training
# Using generalized network creation with explicit parameters
net = create_kepler_net(
    input_dims=1,
    output_dims=4,
    num_hidden_layers=3,
    num_neurons_per_layer=50,
    hidden_activation='tanh'
)
model = dde.Model(data, net)

# Loss weights: [res1, res2, res3, res4, ic_x, ic_y, ic_vx, ic_vy, bc_data_x, bc_data_y]
# ODE residuals (4) weight 1.
# ICs (4) should be enforced strongly, e.g., weight 100.
# Data BCs (2) for observed points, e.g., weight 10.
loss_weights = [1, 1, 1, 1, 100, 100, 100, 100, 10, 10] # 10 loss terms
model.compile("adam", lr=1e-3, loss_weights=loss_weights)

# Define model name and output directory
model_name = "Model1_Cartesian"
output_dir_model1 = f"model1_cartesian_outputs" # Corrected f-string from description
os.makedirs(output_dir_model1, exist_ok=True)


print(f"Starting training for {model_name}...")
# Using 1000 iterations for subtask verification as requested
losshistory, train_state = model.train(iterations=1000, display_every=200)
print("Training finished.")

# Plot and save loss history using common function
plot_loss_history(losshistory, model_name, output_dir=output_dir_model1)

# 6. Plotting and Saving Results (specific to this model)
print(f"Plotting and saving specific results for {model_name}...")
t_plot = np.linspace(t_begin, t_end, 200).reshape(-1, 1)
y_pred_tf = model.predict(t_plot) # y_pred will be [x, y, vx, vy]

x_pred = y_pred_tf[:, 0]
y_pred_p = y_pred_tf[:, 1] # Renamed to avoid conflict with plt.ylabel

# Get analytical/numerical solution for comparison over t_plot
sol_exact = solve_ivp(scipy_ode_system, (t_begin, t_end), Y0, dense_output=True, t_eval=t_plot.flatten(), method='RK45')
x_exact = sol_exact.y[0, :]
y_exact = sol_exact.y[1, :]
vx_exact = sol_exact.y[2, :] # For potential future plots
vy_exact = sol_exact.y[3, :] # For potential future plots

# Plot 1: x(t) and y(t) comparison
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(t_plot, x_pred, label="x_pred (PINN)", color='r', linestyle='--')
plt.plot(t_plot, x_exact, label="x_exact (SciPy)", color='b', linestyle='-')
plt.scatter(observe_t_data, observe_xy_data[:, 0], label="x_data (Training)", color='g', marker='o', s=30)
plt.xlabel("Time t")
plt.ylabel("x position")
plt.legend()
plt.title(f"x(t) Comparison - {model_name}")
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(t_plot, y_pred_p, label=f"y_pred ({model_name})", color='r', linestyle='--')
plt.plot(t_plot, y_exact, label="y_exact (SciPy)", color='b', linestyle='-')
plt.scatter(observe_t_data, observe_xy_data[:, 1], label="y_data (Training)", color='g', marker='o', s=30)
plt.xlabel("Time t")
plt.ylabel("y position")
plt.legend()
plt.title(f"y(t) Comparison - {model_name}")
plt.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(output_dir_model1, f"{model_name}_position_time_comparison.png"))
print(f"Saved position vs time plot to {os.path.join(output_dir_model1, f'{model_name}_position_time_comparison.png')}")


# Plot 2: Trajectory (y vs x) comparison
plt.figure(figsize=(8, 8))
plt.plot(x_pred, y_pred_p, label=f"Predicted Trajectory ({model_name})", color='r', linestyle='--')
plt.plot(x_exact, y_exact, label="Exact Trajectory (SciPy)", color='b', linestyle='-')
plt.scatter(observe_xy_data[:, 0], observe_xy_data[:, 1], label="Training Data Points", color='g', marker='o', s=50)
plt.scatter([x0_val], [y0_val], color='black', marker='X', s=100, label="Start Point")
plt.xlabel("x position")
plt.ylabel("y position")
plt.legend()
plt.title(f"Trajectory Comparison (y vs x) - {model_name}")
plt.axis('equal')
plt.grid(True)
plt.savefig(os.path.join(output_dir_model1, f"{model_name}_trajectory_comparison.png"))
print(f"Saved trajectory plot to {os.path.join(output_dir_model1, f'{model_name}_trajectory_comparison.png')}")


# Save DDE's default loss history and training state files
# Note: isplot=False for non-interactive environments for dde.saveplot
dde.saveplot(losshistory, train_state, issave=True, isplot=False, output_dir=output_dir_model1)
print(f"Saved DDE training data to {output_dir_model1}")

print(f"{model_name} script finished.")

if __name__ == "__main__":
    print(f"Running {model_name}.py as main script.")
    pass
