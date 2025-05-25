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
import numpy as np
import matplotlib.pyplot as plt

# --- 1. Daten vorbereiten ---
def generate_damped_oscillator_data(num_samples=1000,
                                   amplitude=1.0,
                                   decay_constant=0.5,
                                   frequency=2.0,
                                   phase=0.0,
                                   noise_amplitude=0.05):
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
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Shape von X_train: {X_train.shape}")
print(f"Shape von y_train: {y_train.shape}")
print(f"Shape von X_test: {X_test.shape}")
print(f"Shape von y_test: {y_test.shape}")

# %%
# --- 2. Modell erstellen (Anzahl der Schichten und Neuronen) ---

def build_oscillator_model(num_hidden_layers=1,
                           neurons_per_layer=32,
                           activation_function='relu',
                           learning_rate=0.001):
    """
    Erstellt ein Keras-Modell für die Vorhersage eines gedämpften Oszillators.

    Args:
        num_hidden_layers (int): Anzahl der versteckten Schichten.
        neurons_per_layer (int): Anzahl der Neuronen in jeder versteckten Schicht.
        activation_function (str): Name der Aktivierungsfunktion für die versteckten Schichten.
        learning_rate (float): Die Lernrate für den Adam-Optimierer.

    Returns:
        keras.Model: Das kompilierte Keras-Modell.
    """
    model = keras.Sequential()

    # Eingabeschicht (implizit, durch die erste Dense-Schicht definiert)
    # Input-Shape ist (1,), da wir nur die Zeit als Eingabe haben
    model.add(layers.Dense(neurons_per_layer, input_shape=(1,), activation=activation_function))

    # Versteckte Schichten
    for _ in range(num_hidden_layers -1): # Beginnen bei 1, da die erste Schicht bereits hinzugefügt wurde
        model.add(layers.Dense(neurons_per_layer, activation=activation_function))

    # Ausgabeschicht
    # Da wir eine einzelne kontinuierliche Position vorhersagen, ist die Ausgabe ein einzelnes Neuron
    # und keine Aktivierungsfunktion (oder 'linear'), da wir keine Begrenzung der Ausgabe wünschen.
    model.add(layers.Dense(1, activation='linear')) # 'linear' ist die Standardaktivierung und kann weggelassen werden

    # --- 3. Optimierer konfigurieren ---
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

    # --- 4. Modell kompilieren ---
    # Metrik: Mean Absolute Error (MAE) ist gut, um die durchschnittliche absolute Abweichung zu sehen.
    # Mean Squared Error (MSE) ist ebenfalls üblich und sensitiver auf größere Fehler.
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

    return model

# --- Konfiguration der Modellparameter (für Einfachheit und Präzision) ---
# Für "einfachst" und "höchste Präzision" müssen wir ein wenig experimentieren.
# Beginnen wir mit einem sehr einfachen Setup und passen es bei Bedarf an.

# Parameter für die Modellarchitektur
NUM_HIDDEN_LAYERS = 2     # Eine oder zwei versteckte Schichten sind oft ein guter Startpunkt.
                           # Für "einfachst" könnten wir sogar 1 versuchen.
NEURONS_PER_LAYER = 32     # Eine kleinere Anzahl von Neuronen (z.B. 16, 32, 64)
ACTIVATION_FUNCTION = 'relu' # ReLU ist eine gute Standardwahl. Auch 'tanh' könnte funktionieren.
LEARNING_RATE = 0.001      # Eine typische Lernrate für Adam.

# Erstelle das Modell
model = build_oscillator_model(num_hidden_layers=NUM_HIDDEN_LAYERS,
                               neurons_per_layer=NEURONS_PER_LAYER,
                               activation_function=ACTIVATION_FUNCTION,
                               learning_rate=LEARNING_RATE)

# Zeige eine Zusammenfassung des Modells (Parameteranzahl ist hier sichtbar)
model.summary()

# %%
# --- 5. Modell trainieren ---
# Das Training kann eine Weile dauern, abhängig von der Datenmenge und der Modellkomplexität.
# 'epochs' ist die Anzahl der Trainingsdurchläufe über den gesamten Datensatz.
# 'batch_size' ist die Anzahl der Samples, die pro Aktualisierung der Modellgewichte verwendet werden.
# 'validation_split' reserviert einen Teil der Trainingsdaten für die Validierung während des Trainings.
history = model.fit(X_train, y_train,
                    epochs=200,
                    batch_size=32,
                    validation_split=0.2,
                    verbose=1) # verbose=1 zeigt den Trainingsfortschritt an

# --- 6. Modell evaluieren ---
loss, mae = model.evaluate(X_test, y_test, verbose=0)
print(f"\nVerlust (MSE) auf dem Testset: {loss:.4f}")
print(f"Mittlere absolute Abweichung (MAE) auf dem Testset: {mae:.4f}")

# --- Vorhersagen machen und visualisieren ---
y_pred = model.predict(X)

plt.figure(figsize=(12, 6))
plt.scatter(X, y, label='Echte Daten (mit Rauschen)', alpha=0.6, s=10)
plt.plot(X, y_pred, color='red', label='Modellvorhersage', linewidth=2)
plt.title('Vorhersage eines gedämpften harmonischen Oszillators')
plt.xlabel('Zeit')
plt.ylabel('Position')
plt.legend()
plt.grid(True)
plt.show()

# Optional: Trainingsverlauf plotten
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Trainings-Verlust')
plt.plot(history.history['val_loss'], label='Validierungs-Verlust')
plt.title('Modell-Verlust während des Trainings')
plt.xlabel('Epoche')
plt.ylabel('Verlust (MSE)')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Trainings-MAE')
plt.plot(history.history['val_mae'], label='Validierungs-MAE')
plt.title('Modell-MAE während des Trainings')
plt.xlabel('Epoche')
plt.ylabel('MAE')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
