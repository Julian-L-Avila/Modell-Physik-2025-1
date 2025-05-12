#!/bin/bash
# measure_times.sh - mide tiempo de ejecución (user) para Fortran, Python, C y Octave
METRIC_FLAG="%U"
METRIC_NAME="user"
FEXE=diffinita_exe
PYEXE="python3 diffinita.py"
CEXE=diffinita_c
OCTAVE_EXE="octave --no-gui --eval"
OUT=times.csv

# Compilar el programa Fortran
gfortran diffinita.f90 -o $FEXE
if [ ! -x "$FEXE" ]; then
    echo "Error: El ejecutable Fortran $FEXE no se creó correctamente."
    exit 1
fi

# Compilar el programa C
gcc diffinita.c -o $CEXE -lm
if [ ! -x "$CEXE" ]; then
    echo "Error: El ejecutable C $CEXE no se creó correctamente."
    exit 1
fi

# Cabecera del CSV
echo "N,time_f90_$METRIC_NAME,time_py_$METRIC_NAME,time_c_$METRIC_NAME,time_octave_$METRIC_NAME" > "$OUT"

# Barrido de N
#for N in $(seq 5 5 50) $(seq 60 10 100); do
for N in $(seq 10 10 50); do
    # Ejecutar el solver Fortran y capturar tiempo user
    t_user_f90=$(/usr/bin/time --format="$METRIC_FLAG" ./$FEXE $N 2>&1)
    t_user_f90=${t_user_f90//,/.}

    # Ejecutar el solver Python y capturar tiempo user
    t_user_py=$(/usr/bin/time --format="$METRIC_FLAG" $PYEXE $N 2>&1)
    t_user_py=${t_user_py//,/.}

    # Ejecutar el solver C y capturar tiempo user
    t_user_c=$(/usr/bin/time --format="$METRIC_FLAG" ./$CEXE $N 2>&1)
    t_user_c=${t_user_c//,/.}

    # Ejecutar el solver Octave y capturar tiempo user
    t_user_octave=$(/usr/bin/time --format="$METRIC_FLAG" $OCTAVE_EXE "diffinita($N)" 2>&1)
    t_user_octave=${t_user_octave//,/.}

    # Mostrar por pantalla
    printf "N=%4d tiempo_f90=%s s tiempo_py=%s s tiempo_c=%s s tiempo_octave=%s s\n" "$N" "$t_user_f90" "$t_user_py" "$t_user_c" "$t_user_octave"

    # Escribir en CSV
    echo "$N,$t_user_f90,$t_user_py,$t_user_c,$t_user_octave" >> "$OUT"
   # echo "$N,$t_user_f90,$t_user_c" >> "$OUT"
done

echo "Resultados de tiempo almacenados en $OUT (métrica: $METRIC_NAME) para Fortran, Python, C y Octave"
