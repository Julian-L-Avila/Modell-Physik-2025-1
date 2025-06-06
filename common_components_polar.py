import deepxde as dde
import numpy as np
import tensorflow as tf
import os # Though not directly used in this script, good for consistency if other common scripts use it.

# Import from the original Cartesian common_components.py
try:
    from common_components import create_kepler_net, plot_loss_history, \
                                  GM as GM_cartesian, \
                                  x0_val as x0_cart, y0_val as y0_cart, \
                                  vx0_val as vx0_cart, vy0_val as vy0_cart, \
                                  t_begin as t_begin_cart, t_end as t_end_cart
except ImportError as e:
    print(f"Failed to import from common_components.py: {e}")
    print("Ensure common_components.py is in the PYTHONPATH and contains all required definitions.")
    # Fallback for critical constants if common_components is missing (e.g. during isolated testing)
    # This is not ideal for production but helps this script be somewhat self-contained for definition.
    if 'GM_cartesian' not in globals(): GM_cartesian = 1.0
    if 't_begin_cart' not in globals(): t_begin_cart = 0.0
    if 't_end_cart' not in globals(): t_end_cart = 10.0
    if 'x0_cart' not in globals(): x0_cart = 1.0
    if 'y0_cart' not in globals(): y0_cart = 0.0
    if 'vx0_cart' not in globals(): vx0_cart = 0.0
    if 'vy0_cart' not in globals(): vy0_cart = 1.0
    # create_kepler_net and plot_loss_history would still be missing if the import fails.

# Define Constants for Polar Coordinates
GM = GM_cartesian
t_begin = t_begin_cart
t_end = t_end_cart
geom_polar = dde.geometry.TimeDomain(t_begin, t_end)

# Define Initial Conditions (Polar) from Cartesian ICs
r0_pol = np.sqrt(x0_cart**2 + y0_cart**2)
theta0_pol = np.arctan2(y0_cart, x0_cart)

epsilon_r0_numpy = 1e-8 # Epsilon for numpy calculations if r0 is zero
# Ensure r0_stable is not zero for division in IC calculation
# max() is Python's built-in, suitable for these initial numpy/float calculations
r0_stable_numpy = max(r0_pol, epsilon_r0_numpy)

drdt0_pol = (x0_cart * vx0_cart + y0_cart * vy0_cart) / r0_stable_numpy
# For dthetadt0, r0_stable_numpy**2 is fine. If r0_stable_numpy is from max(r0_pol, eps), it won't be zero.
dthetadt0_pol = (x0_cart * vy0_cart - y0_cart * vx0_cart) / (r0_stable_numpy**2)


# Define Boundary Function for ICs in Polar Coordinates
def boundary_initial_polar(t_spatial, on_initial):
    """Checks if a point is on the initial boundary for polar coordinates."""
    return on_initial and dde.utils.isclose(t_spatial[0], t_begin)

# Define ODE System for Polar Coordinates
def kepler_ode_system_polar(t_in, y_vec_polar):
    # y_vec_polar is [r, theta, dr_dt, dtheta_dt] (network outputs)
    r_val, theta_val, dr_dt_val, dtheta_dt_val = y_vec_polar[:, 0:1], y_vec_polar[:, 1:2], y_vec_polar[:, 2:3], y_vec_polar[:, 3:4]

    # Calculate time derivatives of network outputs using dde.grad.jacobian
    # dr_dt_calc is d(r_val)/dt, where r_val is the network's output for r(t)
    dr_dt_calc = dde.grad.jacobian(y_vec_polar, t_in, i=0)
    # dtheta_dt_calc is d(theta_val)/dt
    dtheta_dt_calc = dde.grad.jacobian(y_vec_polar, t_in, i=1)

    # ddr_dt_dt_network is d(dr_dt_val)/dt, where dr_dt_val is the network's output for dr/dt
    ddr_dt_dt_network = dde.grad.jacobian(y_vec_polar, t_in, i=2)
    # ddtheta_dt_dt_network is d(dtheta_dt_val)/dt
    ddtheta_dt_dt_network = dde.grad.jacobian(y_vec_polar, t_in, i=3)

    # Numerical stability constants for TensorFlow operations
    epsilon_tf_r = tf.constant(1e-8, dtype=tf.float32) # For r_val to prevent division by zero or log(0) etc.

    r_stable = tf.maximum(r_val, epsilon_tf_r)
    r_stable_sq = tf.square(r_stable)

    # Residuals:
    # 1. Kinematic: dr/dt (calculated from network's r(t)) must equal dr_dt_val (network's output for dr/dt)
    res1_polar = dr_dt_calc - dr_dt_val
    # 2. Kinematic: dtheta/dt (calculated from network's theta(t)) must equal dtheta_dt_val (network's output for dtheta/dt)
    res2_polar = dtheta_dt_calc - dtheta_dt_val

    # 3. Dynamic equation for r: d^2r/dt^2 = r*(dtheta/dt)^2 - GM/r^2
    term_r_omega_sq = r_val * tf.square(dtheta_dt_val)
    term_gm_r_sq = GM / r_stable_sq
    res3_polar = ddr_dt_dt_network - (term_r_omega_sq - term_gm_r_sq)

    # 4. Dynamic equation for theta: d^2theta/dt^2 = -2*(dr/dt)*(dtheta/dt)/r
    term_angular_accel = (2.0 * dr_dt_val * dtheta_dt_val) / r_stable
    res4_polar = ddtheta_dt_dt_network + term_angular_accel

    return [res1_polar, res2_polar, res3_polar, res4_polar]

if __name__ == "__main__":
    print("common_components_polar.py execution test:")
    print(f"GM (from Cartesian): {GM}")
    print(f"Time domain (from Cartesian): [{t_begin}, {t_end}]")
    print(f"Cartesian ICs (for reference): x0={x0_cart}, y0={y0_cart}, vx0={vx0_cart}, vy0={vy0_cart}")
    print(f"Calculated Polar ICs: r0={r0_pol:.4f}, theta0={theta0_pol:.4f} (rad), drdt0={drdt0_pol:.4f}, dthetadt0={dthetadt0_pol:.4f} (rad/s)")

    try:
        net_polar_test = create_kepler_net(
            input_dims=1,
            output_dims=4,
            num_hidden_layers=2,
            num_neurons_per_layer=[40,40],
            hidden_activation='sigmoid'
        )
        print(f"Created a test polar net of type: {type(net_polar_test)}")
        # print(f"Test polar net layer sizes: {net_polar_test.layer_sizes}") # .layer_sizes might not be a public attribute
    except NameError:
        print("Skipped net creation test as 'create_kepler_net' was not imported (likely common_components.py missing).")


    sample_t_numpy = np.array([[t_begin]])
    print(f"Test boundary_initial_polar(t_begin, True) is: {boundary_initial_polar(sample_t_numpy, True)}")
    sample_t_numpy_end = np.array([[t_end]])
    print(f"Test boundary_initial_polar(t_end, True) is: {boundary_initial_polar(sample_t_numpy_end, True)}")

    print("common_components_polar.py defined successfully.")
    # Full test of kepler_ode_system_polar requires DeepXDE model setup and is deferred to model scripts.
