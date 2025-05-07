#!/usr/bin/env bash
set -e

# measure_times.sh — mide únicamente CPU time en user mode (%U)

METRIC_FLAG="%U"
METRIC_NAME="user"

# Archivos y ejecutables
PY=diffinita.py
M=diffinita.m
C_SRC=diffinita.c;    C_EXE=diffinita_c
F_SRC=diffinita.f90;  F_EXE=diffinita_f90

# Compilación
gcc -O2 -lm $C_SRC -o $C_EXE
gfortran -O2 $F_SRC -o $F_EXE

OUT=times.csv
echo "N,time_py_$METRIC_NAME,time_matlab_$METRIC_NAME,time_c_$METRIC_NAME,time_f90_$METRIC_NAME" > $OUT

measure(){
  local t=$(/usr/bin/time -f "$METRIC_FLAG" "$@" 1>/dev/null 2>&1)
  echo "${t//,/\.}"
}

for N in $(seq 5 5 50); do
  printf "N=%2d  " $N
  t_py=$( measure python3 $PY $N )
  t_mat=$( measure matlab -nodisplay -r "N=$N;run('$M');exit" )
  t_c=$(   measure ./$C_EXE  $N )
  t_f=$(   measure ./$F_EXE  $N )
  printf " py=%6s  m=%6s  c=%6s  f90=%6s\n" "$t_py" "$t_mat" "$t_c" "$t_f"
  echo "$N,$t_py,$t_mat,$t_c,$t_f" >> $OUT
done

echo "¡Listo! resultados en $OUT (metric: $METRIC_NAME)"
