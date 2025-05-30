# Establece el terminal de salida
set terminal png size 800,600
set output 'grafica_tiempos.png'

# Establece el título de la gráfica
set title "Comparación de Tiempos de Ejecución (Fortran vs. C)"

# Etiquetas de los ejes
set xlabel "N"
set ylabel "Tiempo de Usuario (segundos)"

# Activa la leyenda
set key left top

# Especifica el separador de columnas (coma en este caso)
set datafile separator ","

# Grafica los datos desde el archivo CSV
plot 'times.csv' using 1:2 with linespoints title 'Fortran 90', \
     'times.csv' using 1:3 with linespoints title 'C'
