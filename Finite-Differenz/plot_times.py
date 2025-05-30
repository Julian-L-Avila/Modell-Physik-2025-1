import pandas as pd
import matplotlib.pyplot as plt

# Leer los datos del archivo CSV
try:
    data = pd.read_csv('times.csv')
except FileNotFoundError:
    print("Error: El archivo times.csv no se encontró.")
    exit(1)

# Extraer las columnas necesarias
N = data['N']
time_f90 = data['time_f90_user']
time_py = data['time_py_user']
time_c = data['time_c_user']
time_octave = data['time_octave_user']

# Crear la gráfica
plt.figure(figsize=(10, 6))
plt.plot(N, time_f90, marker='o', linestyle='-', label='Fortran 90')
plt.plot(N, time_py, marker='s', linestyle='--', label='Python')
plt.plot(N, time_c, marker='^', linestyle='-.', label='C')
plt.plot(N, time_octave, marker='d', linestyle=':', label='Octave')

# Añadir etiquetas y título
plt.xlabel('Tamaño de la malla (N)')
plt.ylabel('Tiempo de CPU (user) [s]')
plt.title('Tiempo de Ejecución vs. Tamaño de Malla')
plt.grid(True)
plt.legend()

# Guardar la gráfica
plt.savefig('images/plot.pdf')
plt.close()

print("Gráfico guardado como images/plot.pdf")
