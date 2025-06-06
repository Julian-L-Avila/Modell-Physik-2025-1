import deepxde as dde
import numpy as np
import tensorflow as tf
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import os

# Import from common_components
try:
    from common_components import GM, t_begin, t_end, geom, create_kepler_net, \
                                x0_val, y0_val, vx0_val, vy0_val, boundary_initial
except ImportError:
    print("Error: common_components.py not found or an import failed.")
    print("Ensure common_components.py is in the same directory or accessible in PYTHONPATH.")
    exit()

# 1. Define the ODE System for DeepXDE (Identical to Model 1)
def kepler_ode_system(t_in, y_vec):
    x_pos, y_pos, vx_vel, vy_vel = y_vec[:, 0:1], y_vec[:, 1:2], y_vec[:, 2:3], y_vec[:, 3:4]

    dx_dt = dde.grad.jacobian(y_vec, t_in, i=0)
    dy_dt = dde.grad.jacobian(y_vec, t_in, i=1)
    dvx_dt = dde.grad.jacobian(y_vec, t_in, i=2)
    dvy_dt = dde.grad.jacobian(y_vec, t_in, i=3)

    r_squared = x_pos**2 + y_pos**2
    epsilon_tf = tf.constant(1e-8, dtype=y_vec.dtype)

    r = tf.sqrt(tf.maximum(r_squared, epsilon_tf))
    r_cubed = tf.maximum(r**3, epsilon_tf)

    res1 = dx_dt - vx_vel
    res2 = dy_dt - vy_vel
    res3 = dvx_dt + GM * x_pos / r_cubed
    res4 = dvy_dt + GM * y_pos / r_cubed

    return [res1, res2, res3, res4]

# SciPy ODE system for generating comparison solution (not for training)
def scipy_ode_system(t_sci, Y_sci):
    x_pos, y_pos, vx_vel, vy_vel = Y_sci[0], Y_sci[1], Y_sci[2], Y_sci[3]
    r_squared = x_pos**2 + y_pos**2
    epsilon_np = 1e-8
    if r_squared < epsilon_np:
        r_cubed = epsilon_np
    else:
        r_cubed = r_squared * np.sqrt(r_squared)
        r_cubed = np.maximum(r_cubed, epsilon_np)

    dvx_dt_sci = -GM * x_pos / r_cubed
    dvy_dt_sci = -GM * y_pos / r_cubed
    return [vx_vel, vy_vel, dvx_dt_sci, dvy_dt_sci]

# 2. Define DeepXDE ICs
ic_x = dde.icbc.IC(geom, lambda t: x0_val, boundary_initial, component=0)
ic_y = dde.icbc.IC(geom, lambda t: y0_val, boundary_initial, component=1)
ic_vx = dde.icbc.IC(geom, lambda t: vx0_val, boundary_initial, component=2)
ic_vy = dde.icbc.IC(geom, lambda t: vy0_val, boundary_initial, component=3)

bcs = [ic_x, ic_y, ic_vx, ic_vy] # No PointSetBCs for Model 2

# 3. Create TimePDE Data Object
# num_domain: points for ODE residuals.
# num_boundary: points for initial conditions (t_begin).
# num_initial=0 as established in Model 1. ICs are sampled via num_boundary at t_begin.
data = dde.data.TimePDE(
    geom,
    kepler_ode_system,
    bcs,
    num_domain=500,
    num_boundary=2, # Sufficient for point ICs at t_begin, filtered by boundary_initial
    num_initial=0,  # Resolved previous error by setting to 0
    train_distribution="pseudo"
)

# 4. Network, Model Compilation, and Training
net = create_kepler_net()
model = dde.Model(data, net)

# Loss weights: [res1, res2, res3, res4, ic_x, ic_y, ic_vx, ic_vy]
# ODE residuals (4) weight 1.
# ICs (4) should be enforced strongly, e.g., weight 100.
loss_weights = [1, 1, 1, 1, 100, 100, 100, 100]
model.compile("adam", lr=1e-3, loss_weights=loss_weights)

# Create output directory if it doesn't exist
output_dir_name_m2 = "model2_outputs"
if not os.path.exists(output_dir_name_m2):
    os.makedirs(output_dir_name_m2)

print("Starting training for Model 2 (No Data Points)...")
# Iterations: 30000 to 50000. Using 30000 for consistency.
losshistory, train_state = model.train(iterations=30000, display_every=1000)
print("Training finished for Model 2.")

# 5. Plotting and Saving Results
print("Plotting and saving results for Model 2...")
t_plot = np.linspace(t_begin, t_end, 200).reshape(-1, 1)
y_pred_model2_tf = model.predict(t_plot)

x_pred_m2 = y_pred_model2_tf[:, 0]
y_pred_m2_p = y_pred_model2_tf[:, 1] # Renamed to avoid conflict

# Generate numerical solution for comparison
Y0_m2 = [x0_val, y0_val, vx0_val, vy0_val]
sol_exact_m2 = solve_ivp(scipy_ode_system, (t_begin, t_end), Y0_m2, dense_output=True, t_eval=t_plot.flatten(), method='RK45')
x_exact_m2 = sol_exact_m2.y[0, :]
y_exact_m2 = sol_exact_m2.y[1, :]

# Plot 1: x(t) and y(t) comparison
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(t_plot, x_pred_m2, label="x_pred (PINN M2)", color='r', linestyle='--')
plt.plot(t_plot, x_exact_m2, label="x_exact (SciPy)", color='b', linestyle='-')
plt.xlabel("Time t")
plt.ylabel("x position")
plt.legend()
plt.title("x(t) Comparison - Model 2")
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(t_plot, y_pred_m2_p, label="y_pred (PINN M2)", color='r', linestyle='--')
plt.plot(t_plot, y_exact_m2, label="y_exact (SciPy)", color='b', linestyle='-')
plt.xlabel("Time t")
plt.ylabel("y position")
plt.legend()
plt.title("y(t) Comparison - Model 2")
plt.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(output_dir_name_m2, "model2_position_time_comparison.png"))
print(f"Saved position vs time plot to {output_dir_name_m2}/model2_position_time_comparison.png")

# Plot 2: Trajectory (y vs x) comparison
plt.figure(figsize=(8, 8))
plt.plot(x_pred_m2, y_pred_m2_p, label="Predicted Trajectory (PINN M2)", color='r', linestyle='--')
plt.plot(x_exact_m2, y_exact_m2, label="Exact Trajectory (SciPy)", color='b', linestyle='-')
plt.scatter([x0_val], [y0_val], color='black', marker='X', s=100, label="Start Point")
plt.xlabel("x position")
plt.ylabel("y position")
plt.legend()
plt.title("Trajectory Comparison (y vs x) - Model 2")
plt.axis('equal')
plt.grid(True)
plt.savefig(os.path.join(output_dir_name_m2, "model2_trajectory_comparison.png"))
print(f"Saved trajectory plot to {output_dir_name_m2}/model2_trajectory_comparison.png")

# Save loss history and training state
dde.saveplot(losshistory, train_state, issave=True, isplot=False, output_dir=output_dir_name_m2)
print(f"Saved loss history and training state to {output_dir_name_m2}")

print("Model 2 script finished.")

if __name__ == "__main__":
    print("Running kepler_model2.py as main script.")
    pass
