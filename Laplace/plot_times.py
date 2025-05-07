# plot_times.py
import argparse
import pandas as pd
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--input",  default="times.csv")
parser.add_argument("--output", default="images/plot.pdf")
# fijamos user como default
parser.add_argument("--metric", default="user", choices=["real","user","sys"])
args = parser.parse_args()

df = pd.read_csv(args.input)
cols = {
    'py':   f'time_py_{args.metric}',
    'm':    f'time_matlab_{args.metric}',
    'c':    f'time_c_{args.metric}',
    'f90':  f'time_f90_{args.metric}',
}

plt.plot(df['N'], df[cols['py']],  marker='o', label='Python')
plt.plot(df['N'], df[cols['m']],   marker='s', label='MATLAB')
plt.plot(df['N'], df[cols['c']],   marker='^', label='C')
plt.plot(df['N'], df[cols['f90']], marker='d', label='Fortran')

plt.xlabel('Tamaño de malla N')
plt.ylabel(f'Tiempo de ejecución ({args.metric}) [s]')
plt.title(f'Tiempo vs N ({args.metric})')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(args.output)
