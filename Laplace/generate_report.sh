#!/usr/bin/env bash
set -e

# 1) Medir tiempos (user fijo)
./measure_times.sh

# 2) Generar la gráfica
python3 plot_times.py --metric user --output images/plot.pdf

# 3) Crear report.tex con plantilla ajustada
cat > report.tex << 'EOF'
\documentclass{article}
\usepackage[utf8]{inputenc}
\usepackage[english]{babel}
\usepackage{amsmath, amsfonts, amsthm, amssymb, mathtools}
\usepackage{derivative}
\usepackage{hyperref, url}
\usepackage[dvipsnames]{xcolor}
\usepackage{fancyhdr, last page}
\usepackage{siunitx}
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

\title{\textbf{Comparación de rendimientos}\\\small{Modelamiento Físico Computacional}}
\author[1]{Camilo Huertas}
\author[1]{Julian Avila}
\affil[1]{Universidad Distrital Francisco José de Caldas}
\date{07/05/2025}

\pagestyle{fancy}
\fancyhead{}
\fancyfoot{}
\lhead{Modelamiento Físico Computacional}
\rhead{Huertas \& Avila}
\rfoot{Página \thepage\ de \pageref{LastPage}}

\begin{document}

\maketitle
\thispagestyle{fancy}
\hrule

\section{Objetivo}
Se automatizó la medición de tiempo de CPU (user) de cuatro implementaciones (Python, MATLAB, C, Fortran) de un solver de la ecuación de Laplace 1D. Se generó un gráfico Tiempo vs. tamaño de malla \(N\).

\section{Scripts y códigos}
\subsection{Bash de medición}
\begin{verbatim}
$(sed -n '1,200p' measure_times.sh)
\end{verbatim}

\subsection{Generación de la gráfica (Python)}
\begin{verbatim}
$(sed -n '1,200p' plot_times.py)
\end{verbatim}

\section{Resultados}
\begin{figure}[H]
  \centering
  \includegraphics[width=0.8\linewidth]{plot.pdf}
  \caption{Tiempo de CPU (user) vs. \(N\) para cada implementación.}
  \label{fig:times}
\end{figure}

\section{Conclusión}
La cadena de herramientas permite desde la medición hasta la generación automática del informe PDF, comparando el rendimiento de cada lenguaje.

\printbibliography

\end{document}
EOF

# 4) Compilar el PDF
pdflatex report.tex >/dev/null
echo "Informe generado: report.pdf"
