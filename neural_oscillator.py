# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.3.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
import csv

# --- Custom Activation Functions ---
def custom_sechlu(x, rho=1.0):
    """
    Implementiert eine Sigmoid-weighted Linear Unit basierend auf der Formel:
    f(x) = x / (1 + exp(-2x / rho))

    Diese Funktion skaliert die Eingabe x mit einer Sigmoid-ähnlichen Kurve.
    Der Parameter rho steuert die Steilheit der Kurve um den Ursprung.

    Argumente:
        x (tf.Tensor): Der Eingabe-Tensor.
        rho (float): Ein anpassbarer Parameter zur Steuerung der Form.
                     Standardwert ist 1.0.

    Rückgabe:
        tf.Tensor: Der Tensor nach Anwendung der Aktivierungsfunktion.
    """
    # Die Formel kann direkt mit TensorFlow-Operationen umgesetzt werden.
    # tf.sigmoid(y) ist mathematisch äquivalent zu 1 / (1 + exp(-y)).
    # Hier ist y = 2x / rho.
    return x * tf.sigmoid(2 * x / rho)

def custom_cauchylu(x, rho=1.0):
    """
    Implementiert eine Cauchy-basierte Aktivierungsfunktion mit der Formel:
    f(x) = x/2 * (1 + 2/pi * arctan(x/rho))

    Diese Funktion skaliert die Eingabe auf eine komplexe, nicht-lineare Weise.
    Der Parameter rho passt die Breite der zentralen Region an.

    Argumente:
        x (tf.Tensor): Der Eingabe-Tensor.
        rho (float): Ein anpassbarer Parameter zur Steuerung der Form.
                     Standardwert ist 1.0.

    Rückgabe:
        tf.Tensor: Der Tensor nach Anwendung der Aktivierungsfunktion.
    """
    # Wir setzen die Formel Schritt für Schritt um
    
    # 1. Berechne den inneren Teil: x / rho
    scaled_x = x / rho
    
    # 2. Berechne den Arcustangens
    arctan_x = tf.math.atan(scaled_x)
    
    # 3. Berechne den gesamten Skalierungsfaktor
    gating_factor = 0.5 * (1.0 + (2.0 / np.pi) * arctan_x)
    
    # 4. Multipliziere mit der ursprünglichen Eingabe x
    #    (Hinweis: Die Formel wurde leicht umgestellt von x/2 * (...) zu x * (0.5 * (...))
    #    für eine klarere Implementierung, das Ergebnis ist identisch)
    return x * gating_factor

def custom_laplacelu(x, rho=1.0):
    """
    Implementiert eine Laplace-basierte Aktivierungsfunktion mit der Formel:
    f(x, rho) = x/2 * (1 + sgn(x) * (1 - exp(-abs(x)/rho)))

    Diese Funktion skaliert die Eingabe x mit einem Faktor, der von der
    kumulativen Verteilungsfunktion der Laplace-Verteilung inspiriert ist.
    Der Parameter rho steuert die Steilheit der Kurve.

    Argumente:
        x (tf.Tensor): Der Eingabe-Tensor.
        rho (float): Ein anpassbarer Parameter zur Steuerung der Form.
                     Standardwert ist 1.0.

    Rückgabe:
        tf.Tensor: Der Tensor nach Anwendung der Aktivierungsfunktion.
    """
    # Wir setzen die Formel direkt mit TensorFlow-Funktionen um.
    
    # 1. Berechne den Absolutwert von x
    abs_x = tf.abs(x)
    
    # 2. Berechne den exponentiellen Term
    exp_term = tf.exp(-abs_x / rho)
    
    # 3. Berechne den von der Vorzeichenfunktion abhängigen Teil
    sign_term = tf.sign(x) * (1.0 - exp_term)
    
    # 4. Berechne den gesamten Skalierungsfaktor ("Gate")
    gating_factor = 0.5 * (1.0 + sign_term)
    
    # 5. Multipliziere mit der ursprünglichen Eingabe x
    #    (Umstellung von x/2 * (...) zu x * (0.5 * (...)) für Klarheit)
    return x * gating_factor

# It's also good practice to register them with Keras if they are going to be used by string name directly in layers
# However, for this task, we will pass the function objects directly.

# --- 1. Daten vorbereiten ---
def generate_damped_oscillator_data(num_samples=1000,
                                   amplitude=1.0,
                                   decay_constant=0.5,
                                   frequency=2.0,
                                   phase=0.0,
                                   noise_amplitude=0.00):
    """
    Generiert synthetische Daten für einen gedämpften harmonischen Oszillator.

    Args:
        num_samples (int): Anzahl der zu generierenden Datenpunkte.
        amplitude (float): Anfangsamplitude des Oszillators.
        decay_constant (float): Die Abklingkonstante (Gamma).
        frequency (float): Die Winkelfrequenz (Omega).
        phase (float): Die Phasenverschiebung.
        noise_amplitude (float): Amplitude des hinzugefügten Rauschens.

    Returns:
        tuple: Ein Tupel von (time_steps, positions)
    """
    time_steps = np.linspace(0, 10, num_samples) # Zeit von 0 bis 10
    true_positions = amplitude * np.exp(-decay_constant * time_steps) * \
                     np.cos(frequency * time_steps + phase)
    # Rauschen hinzufügen, um das Modell robuster zu machen
    noise = noise_amplitude * np.random.randn(num_samples)
    positions = true_positions + noise
    return time_steps, positions

# Generieren der Daten
time_data, position_data = generate_damped_oscillator_data()

# Daten für Keras vorbereiten (Input muss 2D sein, Output kann 1D sein)
# reshape(-1, 1) stellt sicher, dass es eine Spalte und beliebig viele Zeilen hat
X = time_data.reshape(-1, 1)
y = position_data

# Daten aufteilen in Trainings- und Testsets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Shape von X_train: {X_train.shape}")
print(f"Shape von y_train: {y_train.shape}")
print(f"Shape von X_test: {X_test.shape}")
print(f"Shape von y_test: {y_test.shape}")

# %%
# --- Main Experiment Loop ---
neuron_configurations = [[32, 32], [32, 16], [16, 32]]
# Custom activation functions (custom_sechlu, custom_cauchylu, custom_laplacelu) are already defined globally.

activation_function_map = {
    'relu': tf.keras.activations.relu,
    'gelu': tf.keras.activations.gelu,
    'tanh': tf.keras.activations.tanh,
    # mish requires TensorFlow >= 2.3 (tf.keras.activations.mish) or from tf_addons.
    # Assuming the execution environment supports tf.keras.activations.mish.
    'mish': tf.keras.activations.mish,
    'sechlu': custom_sechlu,
    'cauchylu': custom_cauchylu,
    'laplacelu': custom_laplacelu
}
activation_function_names = list(activation_function_map.keys())

results_list = []

# --- 2. Modell erstellen (Anzahl der Schichten und Neuronen) ---
# Note: build_oscillator_model is defined below this block, which is fine.

def build_oscillator_model(layer1_neurons,
                           layer1_activation_func,
                           layer2_neurons,
                           layer2_activation_func,
                           learning_rate=0.001,
                           input_shape=(1,)):
    """
    Erstellt ein Keras-Modell für die Vorhersage eines gedämpften Oszillators.

    Args:
        layer1_neurons (int): Anzahl der Neuronen in der ersten Schicht.
        layer1_activation_func (callable or str): Aktivierungsfunktion für die erste Schicht.
        layer2_neurons (int): Anzahl der Neuronen in der zweiten Schicht.
        layer2_activation_func (callable or str): Aktivierungsfunktion für die zweite Schicht.
        learning_rate (float): Die Lernrate für den Adam-Optimierer.
        input_shape (tuple): Input-Shape für die erste Schicht.

    Returns:
        keras.Model: Das kompilierte Keras-Modell.
    """
    model = keras.Sequential()

    # Erste Dense Schicht
    model.add(layers.Dense(layer1_neurons, activation=layer1_activation_func, input_shape=input_shape))

    # Zweite Dense Schicht
    model.add(layers.Dense(layer2_neurons, activation=layer2_activation_func))

    # Ausgabeschicht
    model.add(layers.Dense(1, activation='linear'))

    # --- 3. Optimierer konfigurieren ---
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

    # --- 4. Modell kompilieren ---
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mean_absolute_error'])

    return model

# --- Konfiguration der Modellparameter (für Einfachheit und Präzision) ---
# Für "einfachst" und "höchste Präzision" müssen wir ein wenig experimentieren.
# Beginnen wir mit einem sehr einfachen Setup und passen es bei Bedarf an.

# Parameter für die Modellarchitektur
# Diese Werte werden nun direkt beim Aufruf von build_oscillator_model übergeben.
# Beispiel:
# model = build_oscillator_model(layer1_neurons=32, layer1_activation_func='relu',
#                                layer2_neurons=32, layer2_activation_func='relu',
#                                learning_rate=0.001)


# Erstelle das Modell - Beispielaufruf (muss angepasst werden, falls Training direkt hier erfolgen soll)
# Die alten Parameter NUM_HIDDEN_LAYERS, NEURONS_PER_LAYER, ACTIVATION_FUNCTION, LEARNING_RATE
# sind nicht mehr direkt für build_oscillator_model relevant in dieser Form.
# Stattdessen werden die Parameter direkt übergeben.
# Für den Moment kommentieren wir die Modellerstellung und model.summary() aus,
# da die Parameter nicht mehr auf die alte Weise definiert sind.
# model = build_oscillator_model(num_hidden_layers=NUM_HIDDEN_LAYERS,
#                                neurons_per_layer=NEURONS_PER_LAYER,
#                                activation_function=ACTIVATION_FUNCTION,
#                                learning_rate=LEARNING_RATE)

# Zeige eine Zusammenfassung des Modells (Parameteranzahl ist hier sichtbar)
# model.summary()

# %%
# --- Main Experiment Loop (continued) ---
for neurons_config in neuron_configurations:
    layer1_neurons, layer2_neurons = neurons_config
    for act_name1 in activation_function_names:
        for act_name2 in activation_function_names:
            act_func1 = activation_function_map[act_name1]
            act_func2 = activation_function_map[act_name2]

            model_name = f"{layer1_neurons}{act_name1.upper()}{layer2_neurons}{act_name2.upper()}"
            
            print(f"Building and training model: {model_name}...")

            model = build_oscillator_model(
                layer1_neurons=layer1_neurons,
                layer1_activation_func=act_func1,
                layer2_neurons=layer2_neurons,
                layer2_activation_func=act_func2
            )
            
            num_params = model.count_params()
            
            early_stopping_callback = EarlyStopping(
                monitor='val_loss',
                patience=50,  # As per user's original code
                restore_best_weights=True
            )
            
            # Ensure X_train, y_train, X_test, y_test are available from the data prep step
            history = model.fit(
                X_train, y_train,
                epochs=2000, # As per user's original code
                validation_data=(X_test, y_test),
                batch_size=32,
                callbacks=[early_stopping_callback],
                verbose=0  # Keep output clean during the loop
            )
            
            actual_epochs = len(history.history['loss'])
            
            # Evaluate the model on the test set
            loss, mae = model.evaluate(X_test, y_test, verbose=0)
            mse = loss  # model.compile uses 'mean_squared_error' as the loss (already updated)
            
            results_list.append([model_name, num_params, actual_epochs, mae, mse])
            print(f"Completed: {model_name} | Params: {num_params} | Epochs: {actual_epochs} | MAE: {mae:.4f} | MSE: {mse:.4f}")
            
            # Optional: Clear session to free memory if many models are trained
            # tf.keras.backend.clear_session() 
            # Note: clear_session() also clears custom object registry, so use with care if custom objects are not re-registered.
            # For this case, it might be okay as models are built fresh each time.
            # Let's not add clear_session() for now to avoid potential complexities unless memory issues arise.

# --- Save results to TSV ---
output_tsv_file = 'model_comparison_results.tsv'
print(f"Writing results to {output_tsv_file}...")

with open(output_tsv_file, 'w', newline='') as tsvfile:
    writer = csv.writer(tsvfile, delimiter='\t') # Using tab as delimiter
    
    # Write the header
    writer.writerow(['Name of model', 'parameters', 'epochs', 'mae', 'mse'])
    
    # Write the data rows
    for row in results_list: # Assuming results_list is populated by the experiment loop
        writer.writerow(row)
        
print(f"Results successfully saved to {output_tsv_file}.")

# %%
# --- (Old sections for single model training, evaluation, and plotting are now replaced by the loop above) ---
# --- 5. Modell trainieren ---
# ... (code removed / commented out) ...
#
# --- 6. Modell evaluieren ---
# ... (code removed / commented out) ...
#
# --- Vorhersagen machen und visualisieren ---
# ... (code removed / commented out) ...
#
# Optional: Trainingsverlauf plotten
# ... (code removed / commented out) ...
