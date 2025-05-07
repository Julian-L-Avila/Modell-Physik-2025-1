#!/bin/bash
# measure_times.sh - mide tiempo de ejecución (user) para Fortran diffinita.f90
METRIC_FLAG="%U"
METRIC_NAME="user"
FEXE=diffinita_exe
OUT=times.csv

# Compilar el programa
gfortran diffinita.f90 -o $FEXE

# Cabecera del CSV
echo "N,time_f90_$METRIC_NAME" > "$OUT"

# Barrido de N
for N in $(seq 5 5 50) $(seq 60 10 100) $(seq 125 25 200); do
    # Ejecutar el solver con malla N y capturar tiempo user
    # Usar el comando time sin ruta absoluta
    t_user=$(time -f "$METRIC_FLAG" ./$FEXE $N 2>&1 >/dev/null)
    
    # Normalizar coma a punto
    t_user=${t_user//,/.}
    
    # Mostrar por pantalla y escribir en CSV
    printf "N=%4d tiempo=%s s\n" "$N" "$t_user"
    echo "$N,$t_user" >> "$OUT"
done

echo "Resultados almacenados en $OUT (métrica: $METRIC_NAME)"