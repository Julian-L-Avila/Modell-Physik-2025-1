import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

def x_sech_x(x: tf.Tensor) -> tf.Tensor:
    """
    Custom activation function: x / cosh(x).
    Uses tf.math.divide_no_nan to prevent NaN issues.
    """
    return tf.math.divide_no_nan(x, tf.math.cosh(x))

class CustomInitializerFirstLayer(tf.keras.initializers.Initializer):
    def __init__(self):
        super().__init__()
        limit = np.sqrt(6 / 33)
        self.minval = -limit
        self.maxval = limit

    def __call__(self, shape, dtype=None):
        return tf.random.uniform(
            shape, minval=self.minval, maxval=self.maxval, dtype=dtype
        )

    def get_config(self):
        return {}

class CustomInitializerSecondLayer(tf.keras.initializers.Initializer):
    def __init__(self):
        super().__init__()
        limit = np.sqrt(3 / 32)
        self.minval = -limit
        self.maxval = limit

    def __call__(self, shape, dtype=None):
        return tf.random.uniform(
            shape, minval=self.minval, maxval=self.maxval, dtype=dtype
        )

    def get_config(self):
        return {}

def build_model(input_shape=(1,)):
    """Builds the neural network model."""
    model = Sequential([
        Dense(32, activation=x_sech_x, kernel_initializer=CustomInitializerFirstLayer(), input_shape=input_shape),
        Dense(32, activation='tanh', kernel_initializer=CustomInitializerSecondLayer()),
        Dense(1)  # Output layer for position
    ])
    model.compile(optimizer=Adam(), loss='mean_squared_error', metrics=['mae'])
    return model

def generate_damped_oscillator_data(
    amplitude: float = 1.0,
    damping_coefficient: float = 0.1,  # gamma (γ)
    angular_frequency_natural: float = 1.0,  # omega_0 (ω₀)
    phase: float = 0.0,
    t_end: float = 50.0,
    num_points: int = 1000,
):
    """
    Generates data for a damped harmonic oscillator.

    Args:
        amplitude: The initial amplitude of the oscillation.
        damping_coefficient: The damping coefficient (gamma).
        angular_frequency_natural: The natural angular frequency (omega_0).
        phase: The initial phase of the oscillation.
        t_end: The end time for the simulation.
        num_points: The number of data points to generate.

    Returns:
        A tuple containing two numpy arrays: time (t) and position (x).
    """
    t = np.linspace(0, t_end, num_points)

    # Ensure underdamped condition for simplicity as per requirements
    if angular_frequency_natural <= damping_coefficient:
        raise ValueError(
            "Natural frequency must be greater than damping coefficient for underdamped oscillation."
        )

    omega_damped = np.sqrt(angular_frequency_natural**2 - damping_coefficient**2)
    
    x = amplitude * np.exp(-damping_coefficient * t) * np.cos(omega_damped * t + phase)
    
    return t, x

if __name__ == "__main__":
    time_data, position_data = generate_damped_oscillator_data()
    print("Generated time data points:", len(time_data))
    print("Generated position data points:", len(position_data))

    model = build_model()
    model.summary()

    # Define callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=20, min_lr=0.0001)
    callbacks_list = [early_stopping, reduce_lr]
    print("Defined EarlyStopping and ReduceLROnPlateau callbacks.")

    # Reshape time_data for the model
    time_data_reshaped = np.reshape(time_data, (-1, 1))

    # Train the model
    print("Starting model training...")
    history = model.fit(
        time_data_reshaped,
        position_data,
        epochs=1000,
        validation_split=0.2, # Use 20% of data for validation
        callbacks=callbacks_list,
        verbose=1 # You can set verbose to 0 or 2 for less output if preferred
    )
    print("Model training completed.")

    # Generate predictions
    print("Generating predictions...")
    predicted_position_data = model.predict(time_data_reshaped)
    print("Predictions generated.")
