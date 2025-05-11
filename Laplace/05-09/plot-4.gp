set terminal epslatex color colortext standalone size 18cm,10cm
set output 'comparison-04.tex'

set view 60, 30
set size ratio -1
set tmargin 1
set bmargin 1
set lmargin 1
set rmargin 1

set samples 100
set isosamples 100
set xrange [1:2]
set yrange [1:2]

f(x, y) = x * y * log(x * y)

unset key
set xlabel '$x$'
set ylabel '$y$'
set xtics 0.5
set ytics 0.5
set ztics 3
set cbtics 3
set colorbox

set pm3d at s
set palette defined (\
    0    0.001462 0.000466 0.013866, \
    0.1  0.063536 0.028426 0.142378, \
    0.2  0.206756 0.016207 0.31844, \
    0.3  0.388753 0.063532 0.463555, \
    0.4  0.591074 0.126453 0.506151, \
    0.5  0.780317 0.211979 0.444567, \
    0.6  0.926106 0.336377 0.323905, \
    0.7  0.993248 0.553364 0.201238, \
    0.8  0.987053 0.754892 0.117433, \
    0.9  0.974417 0.909231 0.126547, \
    1.0  0.987053 0.991438 0.749504 \
)

set multiplot layout 2,3 rowsfirst title ""

# Titles above top row
set label 1 "Fortran"  at screen 0.17,0.95 center
set label 2 "Python"   at screen 0.5, 0.95 center
set label 3 "Analytic" at screen 0.83,0.95 center

# Top row – 3D surfaces
splot './fortran/data_poisson-04-fortran.txt' using 1:2:3 with pm3d
splot './python/data_poisson-04-python.txt' using 1:2:3 with pm3d
splot f(x, y) with pm3d

# Bottom row – top-down view (color maps)
set view map
splot './fortran/data_poisson-04-fortran.txt' using 1:2:3 with pm3d
splot './python/data_poisson-04-python.txt' using 1:2:3 with pm3d
splot f(x, y) with pm3d

unset multiplot
