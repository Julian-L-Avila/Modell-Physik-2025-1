#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

// Function to calculate the right-hand side of the equation
double ladoDerecho(double x) {
    const double pi = 3.14159265358979323846;
    return -pi * pi * cos(pi * x);
}

// Function to solve the system of equations using the Gauss-Seidel method
void gaussSeidel(double **A, double *x, double *b, int n, int iter_max) {
    int i, j, m;
    double sum;

    for (m = 0; m < iter_max; m++) {
        for (i = 0; i < n; i++) {
            sum = 0.0;
            for (j = 0; j < n; j++) {
                if (i != j) {
                    sum += A[i][j] * x[j];
                }
            }
            x[i] = (b[i] - sum) / A[i][i];
        }
    }
}

int main(int argc, char *argv[]) {
    int N, nn, i;
    double xi, xf, vi, vf, h, R, P, Q, y;
    double **A; // Changed to double**
    double *b, *x;
    char filename[50];

    // Check for the correct number of arguments
    if (argc != 2) {
        printf("Error: Debe pasar N (nodos interiores) como argumento.\n");
        return 1;
    }

    // Read N from the command-line argument
    if (sscanf(argv[1], "%d", &N) != 1 || N < 1) {
        printf("Error: N debe ser un entero positivo.\n");
        return 1;
    }

    nn = N + 2;

    // Set problem parameters
    xi = -1.0;
    xf = 2.0;
    vi = -1.0;
    vf = 1.0;
    h = (xf - xi) / (nn - 1);

    // Allocate memory
    A = (double **)malloc(N * sizeof(double *)); // Allocate memory for the rows
    if (A == NULL && N > 0) {
        printf("Error: No se pudo asignar memoria para la matriz A (rows).\n");
        return 1;
    }
    if (N > 0) {
        for (i = 0; i < N; i++) {
            A[i] = (double *)malloc(N * sizeof(double)); // Allocate memory for the columns
            if (A[i] == NULL) {
                printf("Error: No se pudo asignar memoria para la matriz A (columns).\n");
                // Free previously allocated rows
                for (int j = 0; j < i; j++) {
                    free(A[j]);
                }
                free(A);
                return 1;
            }
        }
    }


    b = (double *)malloc(N * sizeof(double));
    x = (double *)malloc(N * sizeof(double));
    if (N > 0 && (b == NULL || x == NULL)) {
        printf("Error: No se pudo asignar memoria para los vectores b o x.\n");
        // Free previously allocated memory
        if (A != NULL) {
            for (i = 0; i < N; i++) {
                free(A[i]);
            }
            free(A);
        }
        free(b);
        free(x);
        return 1;
    }

    // Initialize A, b, and x
    if (N > 0) {
        for (i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                A[i][j] = 0.0;
            }
            b[i] = 0.0;
            x[i] = 0.0;
        }
    }

    R = 1.0 / (h * h);
    P = -2.0 / (h * h);
    Q = 1.0 / (h * h);

    // First row
    if (N > 0) {
        A[0][0] = P;
        if (N > 1) A[0][1] = Q;
        y = ladoDerecho(xi + h);
        b[0] = y - vi * R;
    }

    // Intermediate rows
    for (i = 1; i < N - 1; i++) {
        A[i][i - 1] = R;
        A[i][i] = P;
        A[i][i + 1] = Q;
        y = ladoDerecho(xi + (i + 1) * h);
        b[i] = y;
    }

    // Last row
    if (N > 1) A[N - 1][N - 2] = R;
    if (N > 0) {
        A[N - 1][N - 1] = P;
        y = ladoDerecho(xi + N * h);
        b[N - 1] = y - vf * Q;
    }

    // Solve the system using Gauss-Seidel
    if (N > 0) {
        gaussSeidel(A, x, b, N, 10000);
    }

    // Save the results to a file
    sprintf(filename, "resultado_%d.dat", N);
    FILE *fp = fopen(filename, "w");
    if (fp == NULL) {
        printf("Error: No se pudo abrir el archivo %s para escritura.\n", filename);
        // Free previously allocated memory
        if (A != NULL) {
            for (i = 0; i < N; i++) {
                free(A[i]);
            }
            free(A);
        }
        free(b);
        free(x);
        return 1;
    }

    fprintf(fp, "%10.5f %10.5f\n", xi, vi);
    for (i = 0; i < N; i++) {
        fprintf(fp, "%10.5f %10.5f\n", xi + (i + 1) * h, x[i]);
    }
    fprintf(fp, "%10.5f %10.5f\n", xf, vf);
    fclose(fp);


    // Free allocated memory
    if (A != NULL) {
        for (i = 0; i < N; i++) {
            free(A[i]);
        }
        free(A);
    }
    free(b);
    free(x);

    return 0;
}
