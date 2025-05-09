#!/usr/bin/env bash
set -e

# 0) Crear directorio de imágenes si no existe
mkdir -p images

# 1) Medir tiempos
echo "Midiendo tiempos de ejecución..."
./measure_times.sh

# 2) Graficar
echo "Generando gráfica..."
python3 plot_times.py --metric user --output images/plot.pdf

# 3) Generar LaTeX
echo "Generando plantilla LaTeX..."
tagfile=informe_laplace_2d_fortran.tex
cat > $tagfile << 'EOF'
\documentclass{article}
\usepackage[utf8]{inputenc}
\usepackage[english]{babel}
\usepackage{amsmath, amsfonts, amsthm, amssymb, mathtools}
\usepackage{derivative}
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
\author[1]{Camilo Huertas \and Julian Avila}
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
Este trabajo presenta un análisis del rendimiento computacional de un solver para la ecuación de Laplace utilizando el método de diferencias finitas. El estudio se centra en la relación entre el tiempo de ejecución y el tamaño de la malla utilizada. Se automatizó la medición del tiempo de CPU (user) para la implementación en Fortran 90.

\section{Metodología}
\subsection{Modelo Matemático}
La ecuación de Laplace en una dimensión es:
\begin{equation}
    \frac{d^2u}{dx^2} = 0
\end{equation}

La discretización por diferencias finitas de esta ecuación conduce a un sistema de ecuaciones lineales de la forma:
\begin{equation}
    \frac{u_{i+1} - 2u_i + u_{i-1}}{h^2} = f(x_i)
\end{equation}
donde $h$ es el tamaño de paso de la malla.

\subsection{Implementación Computacional}
El sistema de ecuaciones se resolvió mediante el método iterativo de Gauss-Seidel implementado en Fortran 90. El dominio considerado fue $x \in [-1, 2]$ con condiciones de frontera $u(-1) = -1$ y $u(2) = 1$.

\section{Scripts y códigos utilizados}
\subsection{Código Fortran (diffinita.f90)}
\begin{verbatim}
$(sed -n '1,60p' diffinita.f90)
\end{verbatim}

\subsection{Script de medición de tiempos (measure\_times.sh)}
\begin{verbatim}
$(sed -n '1,50p' measure_times.sh)
\end{verbatim}

\subsection{Generación de la gráfica (plot\_times.py)}
\begin{verbatim}
$(sed -n '1,30p' plot_times.py)
\end{verbatim}

\section{Resultados}
\begin{figure}[H]
 \centering
 \includegraphics[width=0.8\linewidth]{plot.pdf}
 \caption{Tiempo de CPU (user) vs. tamaño de malla $N$.}
 \label{fig:times}
\end{figure}

\section{Análisis}
Como se observa en la Figura \ref{fig:times}, el tiempo de ejecución muestra una tendencia aproximadamente cuadrática con respecto al tamaño de la malla, lo que concuerda con la complejidad teórica del método de Gauss-Seidel, que es $\mathcal{O}(N^2)$ por iteración, donde $N$ es el número de incógnitas.

Esta relación cuadrática se debe principalmente a dos factores:
\begin{enumerate}
    \item El aumento del número de ecuaciones e incógnitas con el tamaño de la malla.
    \item El costo computacional de cada iteración del método de Gauss-Seidel, que es $\mathcal{O}(N^2)$.
\end{enumerate}

\section{Conclusión}
La cadena de herramientas implementada permite automatizar todo el proceso desde la medición hasta la generación del informe PDF. El estudio demuestra la relación cuadrática entre el tiempo de ejecución y el tamaño de la malla en la resolución de la ecuación de Laplace mediante diferencias finitas y el método de Gauss-Seidel.

Para problemas de mayor dimensión o que requieran mallas más finas, sería recomendable considerar métodos alternativos con mejor escalabilidad, como los métodos multigrid o los métodos de gradiente conjugado precondicionado.

\end{document}
EOF

# 4) Compilar LaTeX
echo "Compilando documento LaTeX..."
pdflatex $tagfile
pdflatex $tagfile  # Segunda pasada para referencias

# 5) Limpiar archivos temporales
echo "Limpiando archivos temporales..."
# Eliminar archivos temporales de LaTeX explícitamente mencionados
rm -f informe_laplace_2d_fortran.aux informe_laplace_2d_fortran.log informe_laplace_2d_fortran.out informe_laplace_2d_fortran.tex 2>/dev/null

# Lista de otras extensiones de archivos temporales de LaTeX
temp_exts=(".toc" ".lof" ".lot" ".bbl" ".bcf" ".blg" ".run.xml" ".synctex.gz")
for ext in "${temp_exts[@]}"; do
    rm -f "informe_laplace_2d_fortran${ext}" 2>/dev/null
done

# Eliminar los archivos de resultados generados por el código Fortran
rm -f resultado_*.dat 2>/dev/null

# Eliminar el ejecutable compilado
rm -f diffinita_exe 2>/dev/null

# Mantener times.csv e imágenes como solicitado
echo "Proceso completado. Informe generado: informe_laplace_2d_fortran.pdf"
echo "Se conservaron: times.csv y el directorio images/"
