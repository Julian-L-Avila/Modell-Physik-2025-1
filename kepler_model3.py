import deepxde as dde
import numpy as np
import tensorflow as tf
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import os

# Import from common_components
try:
    # Ensure all necessary components are imported, including the new ones
    from common_components import GM, t_begin, t_end, geom, \
                                x0_val, y0_val, vx0_val, vy0_val, boundary_initial, \
                                create_kepler_net, plot_loss_history # Updated imports
except ImportError as e:
    print(f"Error importing from common_components.py: {e}")
    print("Ensure common_components.py is in the same directory or accessible in PYTHONPATH, and contains all required functions.")
    exit()

# Calculate E0 as a global TensorFlow constant
# Using values imported from common_components
r0_sq_g = x0_val**2 + y0_val**2
epsilon_g = 1e-8 # Small epsilon for stability in E0 calculation
# Ensure r0_stable_g is not zero if GM is non-zero and r0_sq_g is zero.
# max with epsilon_g**2 for sqrt, then max with epsilon_g for division.
r0_stable_g_sqrt_arg = max(r0_sq_g, epsilon_g**2)
r0_stable_g = np.sqrt(r0_stable_g_sqrt_arg)
E0_denominator = max(r0_stable_g, epsilon_g) # prevent division by zero if r0_stable_g is zero

E0_global_val = 0.5 * (vx0_val**2 + vy0_val**2) - GM / E0_denominator
E0_tf_global = tf.constant(E0_global_val, dtype=tf.float32)


# 1. Define the Energy-Conserving ODE System for DeepXDE
def kepler_energy_ode_system(t_in, y_vec):
    """
    Defines the system for DeepXDE using kinematic relations and energy conservation.
    y_vec corresponds to [x, y, vx, vy].
    t_in is the time coordinate.
    """
    x_pos, y_pos, vx_vel, vy_vel = y_vec[:, 0:1], y_vec[:, 1:2], y_vec[:, 2:3], y_vec[:, 3:4]

    # Kinematic relations: dx/dt = vx, dy/dt = vy
    dx_dt_calc = dde.grad.jacobian(y_vec, t_in, i=0)  # d(x_pos)/d(t_in)
    dy_dt_calc = dde.grad.jacobian(y_vec, t_in, i=1)  # d(y_pos)/d(t_in)

    res1 = dx_dt_calc - vx_vel  # Enforce dx/dt = vx
    res2 = dy_dt_calc - vy_vel  # Enforce dy/dt = vy

    # Energy E = 0.5 * (vx^2 + vy^2) - GM / r
    r_squared = x_pos**2 + y_pos**2
    epsilon_r_sq = tf.constant(1e-8, dtype=y_vec.dtype) # for r_squared before sqrt

    # r_stable must use maximum with epsilon_r_sq to avoid sqrt of tiny number if r_squared is zero
    r_stable = tf.sqrt(tf.maximum(r_squared, epsilon_r_sq))

    # Potential energy, ensure r_stable is not zero if GM/r_stable is computed.
    # tf.maximum on r_stable itself (after sqrt) can prevent division by zero.
    epsilon_r_stable = tf.constant(1e-8, dtype=y_vec.dtype)
    potential_energy = -GM / tf.maximum(r_stable, epsilon_r_stable)
    kinetic_energy = 0.5 * (vx_vel**2 + vy_vel**2)
    E_val = kinetic_energy + potential_energy  # Shape (batch_size, 1)

    # Conservation of energy: dE/dt = 0
    # dde.grad.jacobian(ys, xs, i=0, j=0) computes d ys[:,i] / d xs[:,j]
    # If E_val is (N,1) and t_in is (N,1), we want d E_val[:,0] / d t_in[:,0]
    # which is dde.grad.jacobian(E_val, t_in, i=0, j=0) or simply dde.grad.jacobian(E_val, t_in)
    # dE_dt = dde.grad.jacobian(E_val, t_in)
    # res3 = dE_dt # This should be enforced to be 0
    # MODIFIED: Enforce E(t) - E0 = 0
    res3 = E_val - E0_tf_global

    return [res1, res2, res3]

# 2. Generate Training Data (Numerical Solution using SciPy - standard ODEs)
def scipy_ode_system(t_sci, Y_sci): # Same as in Model 1
    x_pos_s, y_pos_s, vx_vel_s, vy_vel_s = Y_sci[0], Y_sci[1], Y_sci[2], Y_sci[3]
    r_squared_s = x_pos_s**2 + y_pos_s**2
    epsilon_np = 1e-8
    if r_squared_s < epsilon_np:
        r_cubed_s = epsilon_np
    else:
        r_cubed_s = r_squared_s * np.sqrt(r_squared_s)
        r_cubed_s = np.maximum(r_cubed_s, epsilon_np)

    dvx_dt_sci = -GM * x_pos_s / r_cubed_s
    dvy_dt_sci = -GM * y_pos_s / r_cubed_s
    return [vx_vel_s, vy_vel_s, dvx_dt_sci, dvy_dt_sci]

Y0 = [x0_val, y0_val, vx0_val, vy0_val]
num_data_points = 20
t_eval_data = np.linspace(t_begin, t_end, num_data_points)

sol_data = solve_ivp(scipy_ode_system, (t_begin, t_end), Y0, dense_output=True, t_eval=t_eval_data, method='RK45')
observe_t_data = sol_data.t.reshape(-1, 1)
observe_xy_data = sol_data.y[:2, :].T

# 3. Define DeepXDE ICs and Data BCs (Identical to Model 1)
ic_x = dde.icbc.IC(geom, lambda t: x0_val, boundary_initial, component=0)
ic_y = dde.icbc.IC(geom, lambda t: y0_val, boundary_initial, component=1)
ic_vx = dde.icbc.IC(geom, lambda t: vx0_val, boundary_initial, component=2)
ic_vy = dde.icbc.IC(geom, lambda t: vy0_val, boundary_initial, component=3)

bc_data_x = dde.icbc.PointSetBC(observe_t_data, observe_xy_data[:, 0:1], component=0)
bc_data_y = dde.icbc.PointSetBC(observe_t_data, observe_xy_data[:, 1:2], component=1)

bcs = [ic_x, ic_y, ic_vx, ic_vy, bc_data_x, bc_data_y]

# 4. Create TimePDE Data Object
data = dde.data.TimePDE(
    geom,
    kepler_energy_ode_system, # Using the energy-based ODE system
    bcs,
    num_domain=500,
    num_boundary=2,
    num_initial=0, # Using num_initial=0 as established
    train_distribution="pseudo"
)

# 5. Network, Model Compilation, and Training
model_name = "Model3_Cartesian_EnergyE0"
output_dir_model3 = f"{model_name}_outputs"
os.makedirs(output_dir_model3, exist_ok=True)

net = create_kepler_net(
    input_dims=1,
    output_dims=4,
    num_hidden_layers=3,
    num_neurons_per_layer=50,
    hidden_activation='tanh'
)
model = dde.Model(data, net)

# Loss weights: [res_kin_x, res_kin_y, res_E_minus_E0, ic_x, ic_y, ic_vx, ic_vy, bc_data_x, bc_data_y]
# ODE residuals (3): kinematic x, kinematic y, E-E0.
# ICs (4): x0, y0, vx0, vy0.
# Data BCs (2): x_data, y_data.
loss_weights = [1, 1, 10,  100, 100, 100, 100,  10, 10] # Same number of terms as before
model.compile("adam", lr=1e-3, loss_weights=loss_weights)


print(f"Starting training for {model_name} (Conserving E(t) = E0)...")
# Using 1000 iterations for subtask verification
losshistory, train_state = model.train(iterations=1000, display_every=200)
print(f"Training finished for {model_name}.")

# Plot and save loss history using common function
plot_loss_history(losshistory, model_name, output_dir=output_dir_model3)

# 6. Plotting and Saving Results
print(f"Plotting and saving results for {model_name}...")
t_plot = np.linspace(t_begin, t_end, 200).reshape(-1, 1)
y_pred_model3_tf = model.predict(t_plot)

x_pred_m3 = y_pred_model3_tf[:, 0]
y_pred_m3_p = y_pred_model3_tf[:, 1]
vx_pred_m3 = y_pred_model3_tf[:, 2]
vy_pred_m3 = y_pred_model3_tf[:, 3]

# Numerical solution for comparison
sol_exact_m3 = solve_ivp(scipy_ode_system, (t_begin, t_end), Y0, dense_output=True, t_eval=t_plot.flatten(), method='RK45')
x_exact_m3 = sol_exact_m3.y[0, :]
y_exact_m3 = sol_exact_m3.y[1, :]
vx_exact_m3 = sol_exact_m3.y[2, :]
vy_exact_m3 = sol_exact_m3.y[3, :]

# Plot 1: Position vs. Time Comparison
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(t_plot, x_pred_m3, label="x_pred (PINN M3)", color='r', linestyle='--')
plt.plot(t_plot, x_exact_m3, label="x_exact (SciPy)", color='b', linestyle='-')
plt.scatter(observe_t_data, observe_xy_data[:, 0], label="x_data (Training)", color='g', marker='o', s=30)
plt.xlabel("Time t")
plt.ylabel("x position")
plt.legend()
plt.title(f"x(t) Comparison - {model_name}")
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(t_plot, y_pred_m3_p, label=f"y_pred ({model_name})", color='r', linestyle='--')
plt.plot(t_plot, y_exact_m3, label="y_exact (SciPy)", color='b', linestyle='-')
plt.scatter(observe_t_data, observe_xy_data[:, 1], label="y_data (Training)", color='g', marker='o', s=30)
plt.xlabel("Time t")
plt.ylabel("y position")
plt.legend()
plt.title(f"y(t) Comparison - {model_name}")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir_model3, f"{model_name}_position_time_comparison.png"))
print(f"Saved position vs time plot to {os.path.join(output_dir_model3, f'{model_name}_position_time_comparison.png')}")

# Plot 2: Trajectory Comparison
plt.figure(figsize=(8, 8))
plt.plot(x_pred_m3, y_pred_m3_p, label=f"Predicted Trajectory ({model_name})", color='r', linestyle='--')
plt.plot(x_exact_m3, y_exact_m3, label="Exact Trajectory (SciPy)", color='b', linestyle='-')
plt.scatter(observe_xy_data[:, 0], observe_xy_data[:, 1], label="Training Data Points", color='g', marker='o', s=50)
plt.scatter([x0_val], [y0_val], color='black', marker='X', s=100, label="Start Point")
plt.xlabel("x position")
plt.ylabel("y position")
plt.legend()
plt.title(f"Trajectory Comparison (y vs x) - {model_name}")
plt.axis('equal')
plt.grid(True)
plt.savefig(os.path.join(output_dir_model3, f"{model_name}_trajectory_comparison.png"))
print(f"Saved trajectory plot to {os.path.join(output_dir_model3, f'{model_name}_trajectory_comparison.png')}")

# Plot 3: Energy Conservation (E(t) vs E0)
# Calculate E_pred from PINN solution
r_sq_pred_m3 = x_pred_m3**2 + y_pred_m3_p**2
epsilon_plot = 1e-8**2 # Epsilon for plotting, consistent with E0 calc
r_stable_pred_m3 = np.sqrt(np.maximum(r_sq_pred_m3, epsilon_plot))
pot_E_pred_m3 = -GM / np.maximum(r_stable_pred_m3, np.sqrt(epsilon_plot)) # ensure denominator > 0
kin_E_pred_m3 = 0.5 * (vx_pred_m3**2 + vy_pred_m3**2)
E_pred_vals = kin_E_pred_m3 + pot_E_pred_m3


plt.figure(figsize=(10, 6))
plt.plot(t_plot, E_pred_vals, label=f"Predicted Energy E(t) ({model_name})", color='r', linestyle='--')
plt.axhline(y=E0_global_val, color='b', linestyle='-', label=f"Initial Energy E0 = {E0_global_val:.4f}")
plt.xlabel("Time t")
plt.ylabel("Total Energy E")
plt.title(f"Energy Conservation (E(t) vs E0) - {model_name}")
plt.legend()
plt.grid(True)
# Optional: Set y-axis limits for better visualization if E_pred_vals varies significantly
y_min = min(np.min(E_pred_vals), E0_global_val)
y_max = max(np.max(E_pred_vals), E0_global_val)
padding = (y_max - y_min) * 0.1 if (y_max - y_min) > 1e-6 else 0.1
plt.ylim(y_min - padding, y_max + padding)
plt.savefig(os.path.join(output_dir_model3, f"{model_name}_energy_vs_initial_E0.png"))
print(f"Saved energy conservation plot to {os.path.join(output_dir_model3, f'{model_name}_energy_vs_initial_E0.png')}")

# Save DDE logs
dde.saveplot(losshistory, train_state, issave=True, isplot=False, output_dir=output_dir_model3)
print(f"Saved DDE training data to {output_dir_model3}")

print(f"{model_name} script finished.")

if __name__ == "__main__":
    print(f"Running {model_name}.py as main script.")
    pass
