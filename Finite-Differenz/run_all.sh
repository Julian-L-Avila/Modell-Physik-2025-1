#!/bin/bash

tagfile="informe_laplace_2d_fortran.tex"

# 1) Medir tiempos de ejecución (Fortran, Python, C y Octave)
echo "Midiendo tiempos de ejecución..."
./measure_times.sh
if [ ! -f "times.csv" ]; then
    echo "Error: El archivo times.csv no se generó correctamente."
    exit 1
fi
echo "Resultados de tiempo almacenados en times.csv"

# 2) Generar la gráfica usando Python
echo "Generando gráfica..."
python plot_times.py
if [ ! -f "images/plot.pdf" ]; then
    echo "Error: La gráfica images/plot.pdf no se generó correctamente."
    exit 1
fi
echo "Gráfico guardado como images/plot.pdf"

# 3) Generar la plantilla LaTeX
echo "Generando plantilla LaTeX..."
cat > $tagfile << 'EOF'
\documentclass{article}
\usepackage[utf8]{inputenc}
\usepackage[english]{babel}
\usepackage{amsmath, amsfonts, amsthm, amssymb, mathtools}
\usepackage{hyperref, url}
\usepackage[dvipsnames]{xcolor}
\usepackage{fancyhdr, lastpage}
\usepackage[margin=2.54cm]{geometry}
\usepackage{graphicx}
\usepackage[affil-sl]{authblk}
\usepackage{anyfontsize}
\usepackage{graphicx, subcaption}
\usepackage{biblatex}
\usepackage{csquotes}
\usepackage{float}
\usepackage{svg}
\usepackage{tabularx, ragged2e, booktabs}
\usepackage{caption}
\usepackage{cleveref}
\usepackage{tikz}
\usepackage{circuitikz}
\usetikzlibrary{patterns,calc,arrows.meta,shadows,external,decorations.pathmorphing}
\usepackage{pgfplots}
\pgfplotsset{compat=newest}
\usepgfplotslibrary{statistics,fillbetween}
\addbibresource{ref.bib}
\graphicspath{ {images/} }
\hypersetup{ colorlinks=true, linkcolor=black, urlcolor=blue }
\title{\textbf{Análisis de Rendimiento: Solver de Laplace}\\\small{Modelamiento Físico Computacional}}
\author[1]{Camilo Huertas, Julian Avila}
\affil[1]{Universidad Distrital Francisco José de Caldas}
\date{\today}
\pagestyle{fancy}
\fancyhead{}
\fancyfoot{}
\lhead{Modelamiento Físico Computacional}
\rhead{Análisis de Rendimiento}
\rfoot{Página \thepage\ de \pageref{LastPage}}
\begin{document}
\maketitle
\thispagestyle{fancy}
\hrule
\section{Objetivo}
Este trabajo presenta un análisis del rendimiento computacional de un solver para la ecuación de Laplace utilizando el método de diferencias finitas. El estudio se centra en la relación entre el tiempo de ejecución y el tamaño de la malla utilizada. Se automatizó la medición del tiempo de CPU (user) para la implementación en Fortran 90, Python, C y Octave.

\section{Metodología}
\subsection{Modelo Matemático}
La ecuación de Laplace en una dimensión es:
\begin{equation}
    \frac{\mathrm{d}^2 u}{\mathrm{d} x^2} = 0
\end{equation}

La discretización por diferencias finitas de esta ecuación conduce a un sistema de ecuaciones lineales de la forma:
\begin{equation}
    \frac{u_{i+1} - 2u_i + u_{i-1}}{h^2} = f(x_i)
\end{equation}
donde $h$ es el tamaño de paso de la malla.

\subsection{Implementación Computacional}
El sistema de ecuaciones se resolvió mediante el método iterativo de Gauss-Seidel implementado en Fortran 90, Python, C y Octave.

\section{Resultados}
\begin{figure}[H]
 \centering
 \includegraphics[width=0.8\linewidth]{plot.pdf}
 \caption{Tiempo de CPU (user) vs. tamaño de malla $N$ para Fortran, Python, C y Octave.}
 \label{fig:times}
\end{figure}

\section{Análisis}
Como se observa en la Figura \ref{fig:times}, el tiempo de ejecución muestra una tendencia aproximadamente cuadrática con respecto al tamaño de la malla para las cuatro implementaciones, lo que concuerda con la complejidad teórica del método de Gauss-Seidel, que es $\mathcal{O}(N^2)$ por iteración. Se puede observar una diferencia en el tiempo de ejecución entre Fortran, Python, C y Octave, lo cual es esperado debido a la naturaleza de los lenguajes (compilado vs. interpretado). La implementación en C y Fortran muestran un rendimiento superior al de Python y Octave, si una diferencia notable entre los dos.

\subsection{Comparación C y Fortran}
Al ser estos los 2 lenguajes de programación con menor tiempo de ejecución, se grafica una comparación entre los dos.

\begin{figure}[H]
 \centering
 \includegraphics[width=0.8\linewidth]{grafica_tiempos.png}
 \caption{Tiempo de CPU (user) vs. tamaño de malla $N$ para Fortran y C.}
 \label{fig:times}
\end{figure}


\section{Conclusión}
La cadena de herramientas implementada permite automatizar todo el proceso desde la medición hasta la generación del informe PDF para las cuatro implementaciones. La implementación demuestra la relación cuadrática entre el tiempo de ejecución y el tamaño de la malla en la resolución de la ecuación de Laplace mediante diferencias finitas y el método de Gauss-Seidel en Fortran, Python, C y Octave. La comparación entre lenguajes subraya las diferencias en rendimiento que pueden surgir en problemas computacionalmente intensivos, donde los lenguajes compilados como C y Fortran tienden a ser más rápidos que los lenguajes interpretados como Python y Octave.


\end{document}
EOF
echo "Plantilla LaTeX generada: $tagfile"

# 4) Compilar el documento LaTeX
echo "Compilando documento LaTeX..."
pdflatex $tagfile
pdflatex $tagfile # Compilar dos veces para las referencias cruzadas
biber informe_laplace_2d_fortran # Procesar la bibliografía
pdflatex $tagfile # Compilar por tercera vez para la bibliografía y referencias

if [ -f "$tagfile".pdf ]; then
    echo "Documento LaTeX compilado: $tagfile".pdf
else
    echo "Error: La compilación del documento LaTeX falló."
fi

# 5) Limpiar archivos temporales
echo "Limpiando archivos temporales..."

# Eliminar los ejecutables compilados
rm -f diffinita_exe 2>/dev/null
rm -f diffinita_c 2>/dev/null

# Eliminar archivos de resultados generados por Fortran, Python, C y Octave
rm -f resultado_*.dat 2>/dev/null

# Eliminar archivos temporales de LaTeX
rm -f informe_laplace_2d_fortran.aux 2>/dev/null
rm -f informe_laplace_2d_fortran.log 2>/dev/null
rm -f informe_laplace_2d_fortran.out 2>/dev/null
rm -f informe_laplace_2d_fortran.tex 2>/dev/null
rm -f informe_laplace_2d_fortran.bbl 2>/dev/null
rm -f informe_laplace_2d_fortran.blg 2>/dev/null

# Lista de otras extensiones de archivos temporales de LaTeX
temp_exts=(".toc" ".lof" ".lot" ".bcf" ".run.xml" ".synctex.gz")
for ext in "${temp_exts[@]}"; do
    rm -f "informe_laplace_2d_fortran${ext}" 2>/dev/null
done

# Conservar times.csv e imágenes
echo "Se conservaron: times.csv y el directorio images/"

echo "Proceso completado. Informe generado: $tagfile".pdf
