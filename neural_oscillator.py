# --- Third-party Library Imports ---
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt

# --- Custom Activation Function ---
def x_sech_x(x: tf.Tensor) -> tf.Tensor:
    """
    Custom activation function: f(x) = x / cosh(x).
    This function is designed to introduce non-linearity into the network.
    cosh(x) is always >= 1, so direct division by zero is not a concern here,
    but tf.math.divide_no_nan provides robustness for general cases.
    """
    return tf.math.divide_no_nan(x, tf.math.cosh(x))

# --- Custom Weight Initializers ---
class CustomInitializerFirstLayer(tf.keras.initializers.Initializer):
    """
    Custom weight initializer for the first hidden layer.
    Initializes weights uniformly within the range [-sqrt(6/33), sqrt(6/33)].
    This range is derived from Xavier/Glorot initialization principles,
    where the limit is sqrt(6 / (fan_in + fan_out)).
    For the first layer: fan_in=1 (input dimension from time data), fan_out=32 (units in this layer).
    So, limit = sqrt(6 / (1 + 32)) = sqrt(6/33).
    """
    def __init__(self):
        super().__init__()
        # Calculate limits for uniform distribution.
        limit = np.sqrt(6 / (1 + 32)) 
        self.minval = -limit
        self.maxval = limit

    def __call__(self, shape, dtype=None):
        # Returns a tensor of the given shape filled with values from the uniform distribution.
        return tf.random.uniform(
            shape, minval=self.minval, maxval=self.maxval, dtype=dtype
        )

    def get_config(self):
        # Returns a serializable dictionary of arguments used to initialize this initializer.
        return {"minval": self.minval, "maxval": self.maxval}

class CustomInitializerSecondLayer(tf.keras.initializers.Initializer):
    """
    Custom weight initializer for the second hidden layer.
    Initializes weights uniformly within the range [-sqrt(3/32), sqrt(3/32)].
    This range is also based on Xavier/Glorot initialization:
    For the second layer: fan_in=32 (units from previous layer), fan_out=32 (units in this layer).
    So, limit = sqrt(6 / (32 + 32)) = sqrt(6/64) = sqrt(3/32).
    """
    def __init__(self):
        super().__init__()
        # Calculate limits for uniform distribution.
        limit = np.sqrt(6 / (32 + 32)) # Equivalent to np.sqrt(3/32)
        self.minval = -limit
        self.maxval = limit

    def __call__(self, shape, dtype=None):
        return tf.random.uniform(
            shape, minval=self.minval, maxval=self.maxval, dtype=dtype
        )

    def get_config(self):
        return {"minval": self.minval, "maxval": self.maxval}

# --- Model Definition ---
def build_model(input_shape=(1,)):
    """
    Builds and compiles the neural network model for the oscillator.

    Args:
        input_shape (tuple): The shape of the input data (e.g., (1,) for a single time feature).

    Returns:
        tf.keras.Model: The compiled Keras model.
    """
    model = Sequential([
        # First hidden layer:
        # - 32 units (neurons): Determines the layer's capacity.
        # - Custom activation function `x_sech_x`: Introduces non-linearity.
        # - Custom weight initializer `CustomInitializerFirstLayer`: Sets initial weights.
        # - input_shape: Defines the shape of the input (a single feature, time).
        Dense(32, activation=x_sech_x, kernel_initializer=CustomInitializerFirstLayer(), input_shape=input_shape, name="hidden_layer_1"),
        
        # Second hidden layer:
        # - 32 units: Further processes features from the previous layer.
        # - 'tanh' activation function (hyperbolic tangent): Another common non-linear activation.
        # - Custom weight initializer `CustomInitializerSecondLayer`: Sets initial weights.
        Dense(32, activation='tanh', kernel_initializer=CustomInitializerSecondLayer(), name="hidden_layer_2"),
        
        # Output layer:
        # - 1 unit: Predicts the oscillator's position (a single continuous value).
        # - Linear activation (default for Dense layer): Suitable for regression outputs.
        Dense(1, name="output_layer")
    ])
    
    # Compile the model:
    # - Optimizer: Adam (Adaptive Moment Estimation) is an efficient and widely used optimizer.
    #   It adapts learning rates for each parameter.
    # - Loss function: 'mean_squared_error' (MSE) is suitable for regression tasks.
    #   It measures the average squared difference between actual and predicted values, penalizing larger errors more.
    # - Metrics: 'mae' (Mean Absolute Error) provides another measure of prediction accuracy,
    #   representing the average absolute difference between predictions and actuals. It's less sensitive to outliers than MSE.
    model.compile(optimizer=Adam(), loss='mean_squared_error', metrics=['mae'])
    return model

# --- Data Generation Function ---
def generate_damped_oscillator_data(
    amplitude: float = 1.0,
    damping_coefficient: float = 0.1,  # Denoted as gamma (γ) in physics
    angular_frequency_natural: float = 1.0,  # Denoted as omega_0 (ω₀)
    phase: float = 0.0, # Denoted as phi (φ)
    t_end: float = 50.0,
    num_points: int = 1000,
):
    """
    Generates data for an underdamped harmonic oscillator.

    The formula used for an underdamped oscillator is:
    x(t) = A * exp(-γ*t) * cos(ω_d*t + φ)
    where:
        A = amplitude (initial amplitude)
        γ = damping_coefficient (controls the rate of decay)
        ω₀ = angular_frequency_natural (frequency if no damping)
        φ = phase (initial phase shift)
        ω_d = sqrt(ω₀² - γ²) is the damped angular frequency (actual oscillation frequency under damping).

    Args:
        amplitude (float): Initial amplitude of the oscillation.
        damping_coefficient (float): Damping coefficient (γ). Must be less than
                                     angular_frequency_natural for underdamped oscillation.
        angular_frequency_natural (float): Natural angular frequency (ω₀) if there were no damping.
        phase (float): Initial phase (φ) of the oscillation, in radians.
        t_end (float): The end time for the simulation (e.g., 50 seconds).
        num_points (int): The number of data points to generate over the time interval [0, t_end].

    Returns:
        A tuple containing two numpy arrays:
        t (numpy.ndarray): Time points from 0 to t_end.
        x (numpy.ndarray): Position of the oscillator at each corresponding time point.
    
    Raises:
        ValueError: If damping_coefficient is not less than angular_frequency_natural,
                    as this would not result in an underdamped oscillation.
    """
    t = np.linspace(0, t_end, num_points) # Time vector

    # Condition for underdamped oscillation: natural frequency must be greater than damping coefficient.
    # If γ >= ω₀, the system is critically damped or overdamped, and won't oscillate sinusoidally.
    if angular_frequency_natural <= damping_coefficient:
        raise ValueError(
            "For underdamped oscillation, natural frequency (angular_frequency_natural) "
            "must be greater than damping_coefficient."
        )

    # Calculate damped angular frequency (ω_d)
    omega_damped = np.sqrt(angular_frequency_natural**2 - damping_coefficient**2)
    
    # Calculate position x(t) using the damped harmonic oscillator equation
    x = amplitude * np.exp(-damping_coefficient * t) * np.cos(omega_damped * t + phase)
    
    return t, x

# --- Plotting Function ---
def plot_results(time, actual_position, predicted_position, history):
    """
    Plots the results of the model training and prediction.

    Args:
        time (numpy.ndarray): Array of time points.
        actual_position (numpy.ndarray): Array of actual oscillator positions.
        predicted_position (numpy.ndarray): Array of predicted oscillator positions from the model.
        history (tf.keras.callbacks.History): History object from model.fit(), containing training metrics.
    """

    # Plot 1: Comparison of Actual Data and Neural Network Prediction
    # This plot shows how well the model's predictions (red dashed line) align with
    # the true oscillator behavior (blue solid line) over time.
    # Ideal scenario: The two lines overlap closely.
    plt.figure(figsize=(12, 6))
    plt.plot(time, actual_position, label='Actual Data', color='blue', linestyle='-')
    plt.plot(time, predicted_position, label='NN Prediction', color='red', linestyle='--')
    plt.title('Damped Harmonic Oscillator: Actual vs. Prediction')
    plt.xlabel('Time (s)')
    plt.ylabel('Position')
    plt.legend()
    plt.grid(True)
    plt.tight_layout() # Adjusts plot parameters for a tight layout.
    plt.show()

    # Plot 2: Model Training Loss (Mean Squared Error) vs. Epochs
    # This plot displays the model's loss (error, specifically MSE) on the training set
    # and optionally on the validation set for each epoch.
    # Helps to assess if the model is learning effectively and to spot issues like overfitting
    # (where validation loss starts increasing while training loss continues to decrease).
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Training Loss')
    if 'val_loss' in history.history: # Check if validation loss is available
        plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Training Loss vs. Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (Mean Squared Error)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot 3: Model Training Mean Absolute Error (MAE) vs. Epochs
    # This plot shows the MAE on the training and validation sets. MAE is another metric
    # for regression, representing the average absolute difference between predictions and actual values.
    # It gives a more direct sense of the average error magnitude.
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['mae'], label='Training MAE')
    if 'val_mae' in history.history: # Check if validation MAE is available
        plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('Model Training MAE vs. Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Absolute Error')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot 4: Difference between Prediction and Actual Data vs. Time (Prediction Error/Residuals)
    # This plot shows the residuals (prediction - actual) over time. 
    # It can help identify if the model has systematic errors (e.g., consistently 
    # over or under-predicting in certain regions, or if errors correlate with time).
    # Ideally, errors should be small and randomly distributed around zero.
    difference = predicted_position.flatten() - actual_position.flatten() # Ensure both are 1D arrays for subtraction
    plt.figure(figsize=(12, 6))
    plt.plot(time, difference, color='green', label='Prediction Error')
    plt.title('Prediction Error (NN Prediction - Actual Data) vs. Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Position Difference (Error)')
    plt.axhline(0, color='black', linestyle='--', linewidth=0.8) # Add a zero line for reference
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# --- Main Execution Block ---
if __name__ == "__main__":
    # --- 1. Data Generation ---
    # Generate time and position data for the damped harmonic oscillator.
    # These parameters can be adjusted to simulate different oscillator behaviors.
    TIME_END = 50.0
    NUM_POINTS = 1000
    time_data, position_data = generate_damped_oscillator_data(
        amplitude=1.0,
        damping_coefficient=0.1, # γ
        angular_frequency_natural=1.0, # ω₀
        phase=0.0, # φ
        t_end=TIME_END,
        num_points=NUM_POINTS
    )
    print(f"Generated {len(time_data)} time data points and {len(position_data)} position data points "
          f"for t = 0 to {TIME_END}s.")

    # --- 2. Model Building ---
    # Build the neural network model with the defined architecture and custom components.
    # input_shape=(1,) because our input is a single feature (time).
    model = build_model(input_shape=(1,))
    # Display a summary of the model's architecture (layers, output shapes, number of parameters).
    print("\nModel Summary:")
    model.summary()

    # --- 3. Callbacks Definition ---
    # Callbacks are functions applied at certain stages of the training process (e.g., end of epoch).
    
    # EarlyStopping: Stops training when a monitored metric has stopped improving.
    # - monitor='val_loss': Metric to monitor (validation loss).
    # - patience=50: Number of epochs with no improvement in 'val_loss' after which training will be stopped.
    # - restore_best_weights=True: If training stops due to patience, model weights are reverted
    #   to those from the epoch with the best 'val_loss'.
    early_stopping = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)
    
    # ReduceLROnPlateau: Reduces the learning rate when 'val_loss' has stopped improving.
    # This can help the model make finer adjustments and converge more effectively if it plateaus.
    # - monitor='val_loss': Metric to monitor.
    # - factor=0.2: Factor by which the learning rate will be reduced (new_lr = lr * factor).
    # - patience=20: Number of epochs with no improvement after which learning rate will be reduced.
    # - min_lr=0.0001: Lower bound on the learning rate; it won't be reduced below this value.
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=20, min_lr=0.0001)
    
    callbacks_list = [early_stopping, reduce_lr]
    print("\nDefined EarlyStopping (patience=50) and ReduceLROnPlateau (patience=20) callbacks.")

    # Reshape time_data for the model input: Keras expects input data to have a shape of (samples, features).
    # Our time_data is currently (NUM_POINTS,), needs to be (NUM_POINTS, 1).
    time_data_reshaped = np.reshape(time_data, (-1, 1))

    # --- 4. Model Training ---
    EPOCHS = 1000 # Maximum number of epochs for training.
    VALIDATION_SPLIT = 0.2 # Fraction of training data to use for validation.
    print(f"\nStarting model training for up to {EPOCHS} epochs...")
    
    # Train the model using the generated data. The 'history' object will store training metrics.
    history = model.fit(
        time_data_reshaped, # Input data (time, reshaped)
        position_data,      # Target data (oscillator position)
        epochs=EPOCHS,      # Maximum number of training cycles through the entire dataset.
                            # EarlyStopping might stop training sooner if val_loss plateaus.
        validation_split=VALIDATION_SPLIT, # Fraction of the training data set aside for validation (20%).
                                           # This data is not used for training weights but for evaluating
                                           # loss and metrics at the end of each epoch. It helps monitor
                                           # for overfitting to the training data.
        callbacks=callbacks_list, # List of callbacks to apply during training.
        verbose=1             # Verbosity mode: 1 = progress bar during training.
    )
    print("Model training completed.")
    final_epoch_count = len(history.history['loss'])
    print(f"Training finished after {final_epoch_count} epochs.")


    # --- 5. Prediction Generation ---
    # Use the trained model to make predictions on the original time data (reshaped).
    print("\nGenerating predictions using the trained model...")
    predicted_position_data = model.predict(time_data_reshaped)
    print("Predictions generated.")

    # --- 6. Plotting Results ---
    # Visualize the training process and the model's performance by comparing
    # actual data with predictions and showing training history.
    print("\nPlotting results...")
    plot_results(time_data, position_data, predicted_position_data, history)
    print("\nScript execution finished.")
