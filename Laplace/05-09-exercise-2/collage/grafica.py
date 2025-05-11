import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define file patterns and labels for cases 01, 02 and 03
cases = ['01', '02', '03']
columns = ['Python', 'Fortran', 'Analytical']
file_patterns = {
    'Python': '{case}-python.dat',
    'Fortran': '{case}-fortran.txt',
    'Analytical': '{case}-analitica.dat'
}

# Create figure with 3 rows and 3 columns of 3D subplots
fig = plt.figure(figsize=(15, 15))

# Loop over each case (row)
for i, case in enumerate(cases):
    # Loop over each column type
    for j, col in enumerate(columns):
        index = i * 3 + j + 1
        ax = fig.add_subplot(3, 3, index, projection='3d')
        # Construct filename
        fname = file_patterns[col].format(case=case)
        # Load data: expect columns x, y, V
        data = np.loadtxt(fname)
        x = data[:, 0]
        y = data[:, 1]
        V = data[:, 2]
        # Plot 3D surface using scatter
        ax.plot_trisurf(x, y, V, cmap='viridis', edgecolor='none')
        # Labeling
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('V')
        if i == 0:
            ax.set_title(col)
        if j == 0:
            ax.text2D(-0.3, 0.5, f"Case {case}", transform=ax.transAxes, rotation='vertical', va='center')

plt.tight_layout()
plt.savefig("multiplot_output.pdf")
