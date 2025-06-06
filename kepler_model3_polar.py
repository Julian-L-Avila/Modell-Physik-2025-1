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
        boundary_initial_polar
        # Note: kepler_ode_system_polar is NOT used; a new one is defined below.
    )
except ImportError as e:
    print(f"CRITICAL IMPORT ERROR from common_components_polar: {e}")
    exit()

GM = GM_cartesian # Consistent GM

# --- Calculate Initial Energy E0_polar (as TensorFlow constant) ---
epsilon_init_r_numpy = 1e-8 # For numpy calculation if r0_pol is zero
r0_pol_stable_numpy = max(r0_pol, epsilon_init_r_numpy)

# E = 0.5 * (dr/dt)^2 + 0.5 * (r * dtheta/dt)^2 - GM/r
# Term (r * dtheta/dt) is tangential velocity.
E0_polar_val = 0.5 * (drdt0_pol**2 + (r0_pol_stable_numpy * dthetadt0_pol)**2) - GM / r0_pol_stable_numpy
E0_polar_tf = tf.constant(E0_polar_val, dtype=tf.float32)

# --- Define Energy-Conserving ODE System for Polar Coordinates ---
def kepler_energy_ode_system_polar(t_in, y_vec_polar):
    # y_vec_polar is [r, theta, dr_dt_val (vr), dtheta_dt_val (omega)]
    r_val, theta_val, vr_val, omega_val = y_vec_polar[:, 0:1], y_vec_polar[:, 1:2], y_vec_polar[:, 2:3], y_vec_polar[:, 3:4]

    # Kinematic relations: dr/dt = vr_val and dtheta/dt = omega_val
    # dr_dt_calc is d(r_val)/dt from network's r(t)
    dr_dt_calc = dde.grad.jacobian(y_vec_polar, t_in, i=0)
    # dtheta_dt_calc is d(theta_val)/dt from network's theta(t)
    dtheta_dt_calc = dde.grad.jacobian(y_vec_polar, t_in, i=1)

    res1_kin_r = dr_dt_calc - vr_val      # Enforces that network's r(t) derivative matches network's vr_val
    res2_kin_theta = dtheta_dt_calc - omega_val # Enforces that network's theta(t) derivative matches network's omega_val

    # Energy conservation: E(t) - E0 = 0
    epsilon_r_tf = tf.constant(1e-8, dtype=tf.float32) # For stability if r_val is close to zero
    r_stable_tf = tf.maximum(r_val, epsilon_r_tf) # Use for terms like 1/r

    potential_energy = -GM / r_stable_tf
    # Kinetic energy: 0.5 * m * (vr^2 + (r*omega)^2). Here m=1.
    # Using r_val directly for r*omega, assuming network learns r_val >= 0.
    # If r_val can be negative, tf.abs(r_val) or tf.square(r_val) for r^2 omega^2 might be needed.
    # For now, r_val * omega_val is (tangential velocity component)
    # We use vr_val and omega_val (direct outputs for velocities) for energy.
    kinetic_energy = 0.5 * (tf.square(vr_val) + tf.square(r_val * omega_val))
    E_val_current = kinetic_energy + potential_energy

    res3_energy = E_val_current - E0_polar_tf # Enforces E(t) = E0

    return [res1_kin_r, res2_kin_theta, res3_energy]

# --- Helper for Cartesian solution (for training data generation & comparison) ---
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
sol_cartesian_data = solve_ivp(scipy_ode_system_cartesian, (t_begin, t_end), Y0_cartesian, dense_output=True, t_eval=t_eval_data, rtol=1e-8, atol=1e-8)
observe_t_data = sol_cartesian_data.t.reshape(-1, 1)
x_data_cart = sol_cartesian_data.y[0, :]
y_data_cart = sol_cartesian_data.y[1, :]
r_data_polar = np.sqrt(x_data_cart**2 + y_data_cart**2)
theta_data_polar = np.unwrap(np.arctan2(y_data_cart, x_data_cart))
observe_rtheta_data = np.vstack((r_data_polar, theta_data_polar)).T

# --- DeepXDE ICs and Data BCs ---
ic_r = dde.icbc.IC(geom_polar, lambda t: r0_pol, boundary_initial_polar, component=0)
ic_theta = dde.icbc.IC(geom_polar, lambda t: theta0_pol, boundary_initial_polar, component=1)
ic_drdt = dde.icbc.IC(geom_polar, lambda t: drdt0_pol, boundary_initial_polar, component=2) # For vr
ic_dthetadt = dde.icbc.IC(geom_polar, lambda t: dthetadt0_pol, boundary_initial_polar, component=3) # For omega

bc_data_r = dde.icbc.PointSetBC(observe_t_data, observe_rtheta_data[:, 0:1], component=0)
bc_data_theta = dde.icbc.PointSetBC(observe_t_data, observe_rtheta_data[:, 1:2], component=1)
bcs_polar_m3 = [ic_r, ic_theta, ic_drdt, ic_dthetadt, bc_data_r, bc_data_theta]

# --- TimePDE Data Object ---
data_polar_m3 = dde.data.TimePDE(
    geom_polar,
    kepler_energy_ode_system_polar, # Using the new energy-based ODE system
    bcs_polar_m3,
    num_domain=500,
    num_boundary=20, # Sampling for ICs
    num_initial=0,   # ICs handled by num_boundary
    train_distribution="pseudo"
)

# --- Network, Model Compilation, and Training ---
model_name = "Model3_Polar_EnergyE0"
output_dir_model3_polar = f"{model_name}_outputs"
os.makedirs(output_dir_model3_polar, exist_ok=True)

net_polar_m3 = create_kepler_net(
    input_dims=1,
    output_dims=4, # r, theta, vr, omega
    num_hidden_layers=3,
    num_neurons_per_layer=64,
    hidden_activation='tanh'
)
model_polar_m3 = dde.Model(data_polar_m3, net_polar_m3)

# Loss weights: 3 ODE res (kin_r, kin_theta, E-E0), 4 ICs, 2 Data BCs = 9 components
loss_weights_polar_m3 = [1, 1, 10,  100, 100, 100, 100,  10, 10]
model_polar_m3.compile("adam", lr=1e-3, loss_weights=loss_weights_polar_m3)

iterations = 1000 # Reduced for subtask verification
display_every = 200
print(f"Starting training for {model_name}...")
losshistory, train_state = model_polar_m3.train(iterations=iterations, display_every=display_every)
print(f"Training finished for {model_name}.")

# --- Plotting and Saving Results ---
plot_loss_history(losshistory, model_name, output_dir=output_dir_model3_polar)
dde.saveplot(losshistory, train_state, issave=True, isplot=False, output_dir=output_dir_model3_polar)

t_plot = np.linspace(t_begin, t_end, 200)
y_pred_polar_plot_m3 = model_polar_m3.predict(t_plot.reshape(-1, 1))
r_pred_m3 = y_pred_polar_plot_m3[:, 0]
theta_pred_m3 = y_pred_polar_plot_m3[:, 1]
drdt_pred_m3 = y_pred_polar_plot_m3[:, 2]       # vr_pred
dthetadt_pred_m3 = y_pred_polar_plot_m3[:, 3]   # omega_pred

# Exact Cartesian solution for comparison
Y0_cart_ic_plot = [x0_cart_ic, y0_cart_ic, vx0_cart_ic, vy0_cart_ic]
sol_cart_exact_plot_m3 = solve_ivp(scipy_ode_system_cartesian, (t_begin, t_end), Y0_cart_ic_plot, dense_output=True, t_eval=t_plot, rtol=1e-8, atol=1e-8)
x_exact_cart_plot_m3 = sol_cart_exact_plot_m3.y[0, :]
y_exact_cart_plot_m3 = sol_cart_exact_plot_m3.y[1, :]
# Convert exact Cartesian to Polar for r(t), theta(t) comparison
r_exact_polar_plot_m3 = np.sqrt(x_exact_cart_plot_m3**2 + y_exact_cart_plot_m3**2)
theta_exact_polar_plot_m3 = np.unwrap(np.arctan2(y_exact_cart_plot_m3, x_exact_cart_plot_m3))

# Plot r(t) and theta(t)
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(t_plot, r_pred_m3, 'r--', label='r_pred (PINN)')
plt.plot(t_plot, r_exact_polar_plot_m3, 'b-', label='r_exact (Numerical)')
plt.scatter(observe_t_data.flatten(), r_data_polar, c='k', marker='x', s=50, label='r_data (Training)')
plt.xlabel('Time (t)')
plt.ylabel('r(t)')
plt.legend()
plt.title(f'{model_name} - r(t) Comparison')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(t_plot, theta_pred_m3, 'r--', label='$\\theta_{pred}$ (PINN)')
plt.plot(t_plot, theta_exact_polar_plot_m3, 'b-', label='$\\theta_{exact}$ (Numerical)')
plt.scatter(observe_t_data.flatten(), theta_data_polar, c='k', marker='x', s=50, label='$\\theta_{data}$ (Training)')
plt.xlabel('Time (t)')
plt.ylabel('$\\theta(t)$ (rad)')
plt.legend()
plt.title(f'{model_name} - $\\theta(t)$ Comparison')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir_model3_polar, f"{model_name}_polar_vars_time_comparison.png"))
plt.close()

# Plot trajectory (convert PINN polar to Cartesian)
x_pred_cart_m3 = r_pred_m3 * np.cos(theta_pred_m3)
y_pred_cart_m3 = r_pred_m3 * np.sin(theta_pred_m3)

plt.figure(figsize=(8, 8))
plt.plot(x_pred_cart_m3, y_pred_cart_m3, 'r--', label='Trajectory_pred (PINN Polar)')
plt.plot(x_exact_cart_plot_m3, y_exact_cart_plot_m3, 'b-', label='Trajectory_exact (Numerical Cartesian)')
plt.scatter(x_data_cart, y_data_cart, c='k', marker='x', s=50, label='Training Data (Cartesian Conv.)')
plt.xlabel('x')
plt.ylabel('y')
plt.axis('equal')
plt.legend()
plt.title(f'{model_name} - Trajectory Comparison (x-y)')
plt.grid(True)
plt.savefig(os.path.join(output_dir_model3_polar, f"{model_name}_trajectory_comparison_xy.png"))
plt.close()

# Plot Energy Conservation E(t) vs E0
# Use network outputs for velocities (drdt_pred_m3, dthetadt_pred_m3)
# and r_pred_m3 for r in (r*omega)^2 term and 1/r term.
E_pred_vals_m3 = 0.5 * (drdt_pred_m3**2 + (r_pred_m3 * dthetadt_pred_m3)**2) - GM / np.maximum(r_pred_m3, epsilon_init_r_numpy)
plt.figure(figsize=(10, 6))
plt.plot(t_plot, E_pred_vals_m3, 'r--', label='E_pred (PINN)')
plt.axhline(E0_polar_val, color='b', linestyle='-', label=f'E0_initial = {E0_polar_val:.4f}')
plt.xlabel('Time (t)')
plt.ylabel('Total Energy E(t)')
plt.title(f'{model_name} - Energy Conservation (E(t) vs E0)')
plt.legend()
plt.grid(True)
# Optional: Set y-axis limits for better visualization
y_min_plot = min(np.min(E_pred_vals_m3), E0_polar_val)
y_max_plot = max(np.max(E_pred_vals_m3), E0_polar_val)
padding_plot = (y_max_plot - y_min_plot) * 0.1 if (y_max_plot - y_min_plot) > 1e-6 else 0.1
plt.ylim(y_min_plot - padding_plot, y_max_plot + padding_plot)
plt.savefig(os.path.join(output_dir_model3_polar, f"{model_name}_energy_vs_initial_E0.png"))
plt.close()

print(f"Finished training and plotting for {model_name}. Outputs in {output_dir_model3_polar}/")
print(f"To see plots, check the directory: {output_dir_model3_polar}")

if __name__ == "__main__":
    print(f"Executing {model_name} script...")
    pass
