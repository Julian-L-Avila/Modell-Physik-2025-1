# Gnuplot script to plot numerical and analytical solutions of the Laplace equation
# This version includes color mapping (pm3d) for the 3D surface plots.

# Set the terminal to generate a PDF output
set terminal pdfcairo enhanced color font 'Arial,12'
# Set the output file name
set output 'poisson_solution_color.pdf'

# Set the title for the entire plot (optional, can also title individual splots)
set title 'Numerical vs Analytical Solution of Laplace Equation'

# Set labels for the axes
set xlabel 'x'
set ylabel 'y'
set zlabel 'Potential (V)'

# Set the view angle for the 3D plot
set view 60, 30 # Adjust these angles as needed

# Set the range for the axes (optional, gnuplot can auto-range)
# set xrange [0:1]
# set yrange [0:1]
# set zrange [0:200] # Adjust based on your expected potential range

# Set the grid for better visualization (optional with pm3d)
# set grid

# Enable pm3d for color mapping based on Z-value
set pm3d # 'at b' draws the surface at the bottom of the z-range
set palette # Use the default color palette (you can customize this)

# Use multiplot to place multiple plots on one page
set multiplot layout 1, 2 # Arrange plots in 1 row, 2 columns

# --- Plotting the Numerical Solution ---
set title 'Numerical Solution'
# Plot the surface with color mapping using pm3d
splot 'data_laplace_rect.txt' using 1:2:3 with pm3d notitle

# --- Plotting the Analytical Solution ---
set title 'Analytical Solution'
# Plot the surface with color mapping using pm3d
splot 'data_laplace_rect.txt' using 1:2:4 with pm3d notitle

# End multiplot mode
unset multiplot

# Close the output file
set output

