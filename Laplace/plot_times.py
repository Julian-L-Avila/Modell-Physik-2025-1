#!/usr/bin/env python3
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import os

parser = argparse.ArgumentParser()
parser.add_argument("--input", default="times.csv")
parser.add_argument("--output", default="images/plot.pdf")
parser.add_argument("--metric", default="user", choices=["real","user","sys"])
args = parser.parse_args()

# Asegurar que exista el directorio de imágenes
os.makedirs(os.path.dirname(args.output), exist_ok=True)

# Leer datos
df = pd.read_csv(args.input)

# Construir lista de lenguajes presentes en el CSV header
lang_keys = [col.split('_')[1] for col in df.columns if col.startswith('time_')]
if not lang_keys:  # Si no hay columnas con prefijo time_, usar todas excepto N
    lang_keys = [col for col in df.columns if col != 'N']
    
labels = {'py':'Python','matlab':'MATLAB','c':'C','f90':'Fortran'}

plt.figure(figsize=(10, 6))
for lang in lang_keys:
    col_name = f'time_{lang}_{args.metric}' if f'time_{lang}_{args.metric}' in df.columns else lang
    plt.plot(df['N'], df[col_name], marker='o', label=labels.get(lang, lang))

plt.xlabel('Tamaño de malla N')
plt.ylabel(f'Tiempo de ejecución ({args.metric}) [s]')
plt.title(f'Tiempo vs N ({args.metric})')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(args.output)
print(f"Gráfico guardado como {args.output}")