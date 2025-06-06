import deepxde as dde
import numpy as np
import tensorflow as tf

# Gravitational constant * Central Mass
GM = 1.0

# Time domain
t_begin = 0.0
t_end = 10.0
geom = dde.geometry.TimeDomain(t_begin, t_end)

# Neural Network Architecture
def create_kepler_net():
    """Creates the neural network for the Kepler problem."""
    layer_size = [1] + [50] * 3 + [4]
    activation = "tanh"
    initializer = "Glorot uniform"
    net = dde.nn.FNN(layer_size, activation, initializer)
    return net

# Initial Conditions
x0_val = 1.0
y0_val = 0.0
vx0_val = 0.0
vy0_val = 1.0

# Boundary condition function for initial conditions
def boundary_initial(t, on_initial):
    """
    Defines the boundary for initial conditions.
    Returns True if t is close to t_begin and on_initial is True.
    """
    return on_initial and dde.utils.isclose(t[0], t_begin)

if __name__ == "__main__":
    print(f"GM: {GM}")
    print(f"Time domain: [{t_begin}, {t_end}]")

    # Test neural network creation
    print("Creating and summarizing the Kepler net:")
    try:
        net = create_kepler_net()
        # The summary() method is part of tf.keras.Model, FNN is a subclass
        net.summary()
    except Exception as e:
        print(f"Error during net creation or summary: {e}")

    print(f"Initial conditions: x0={x0_val}, y0={y0_val}, vx0={vx0_val}, vy0={vy0_val}")

    # Test boundary_initial function
    print("\nTesting boundary_initial function:")
    # Test case 1: t is t_begin, on_initial is True
    sample_t_on_initial = np.array([[t_begin]])
    print(f"boundary_initial(t_begin, True) is: {boundary_initial(sample_t_on_initial, True)}")

    # Test case 2: t is t_end, on_initial is False (should be False)
    sample_t_not_on_initial_false = np.array([[t_end]])
    print(f"boundary_initial(t_end, False) is: {boundary_initial(sample_t_not_on_initial_false, False)}")

    # Test case 3: t is t_end, on_initial is True (should be False because t is not t_begin)
    sample_t_not_on_initial_true = np.array([[t_end]])
    print(f"boundary_initial(t_end, True) is: {boundary_initial(sample_t_not_on_initial_true, True)}")

    # Test case 4: t is t_begin, on_initial is False (should be False)
    print(f"boundary_initial(t_begin, False) is: {boundary_initial(sample_t_on_initial, False)}")

    # Test with a value slightly off t_begin but within isclose tolerance
    epsilon = 1e-9
    sample_t_close_to_begin = np.array([[t_begin + epsilon]])
    print(f"boundary_initial(t_begin + epsilon, True) is: {boundary_initial(sample_t_close_to_begin, True)}")

    # Test with a value far from t_begin
    sample_t_far_from_begin = np.array([[t_begin + 1.0]])
    print(f"boundary_initial(t_begin + 1.0, True) is: {boundary_initial(sample_t_far_from_begin, True)}")
