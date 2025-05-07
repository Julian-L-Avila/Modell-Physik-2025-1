#include <math.h>
#include <stdio.h>

#define PARTICION 11  // Tamaño de la partición
#define N 9           // Número de incógnitas
#define VISUALIZA 1   // (0) No visualiza la salida, otro valor la visualiza

// Lado derecho de la ecuación diferencial parcial
double LadoDerecho(double x)
{
    double pi = 3.1415926535897932384626433832;
    return -pi * pi * cos(pi * x);
}

// Resuelve Ax=b usando el método Jacobi
void Jacobi(double A[N][N], double x[], double b[], int n, int iter)
{
    int i, j, m;
    double sum;
    double xt[N] = {0};  // Inicializar a cero

    for (m = 0; m < iter; m++) {
        for (i = 0; i < n; i++) {
            sum = 0.0;
            for (j = 0; j < n; j++) {
                if (i == j) continue;
                sum += A[i][j] * x[j];
            }
            if (A[i][i] == 0.0) return;
            xt[i] = (b[i] - sum) / A[i][i];
        }
        for (i = 0; i < n; i++)
            x[i] = xt[i];
    }
}

// Resuelve Ax=b usando el método Gauss-Seidel
void Gauss_Seidel(double A[N][N], double x[], double b[], int n, int iter)
{
    int i, j, m;
    double sum;

    for (m = 0; m < iter; m++) {
        for (i = 0; i < n; i++) {
            sum = 0.0;
            for (j = 0; j < n; j++) {
                if (i == j) continue;
                sum += A[i][j] * x[j];
            }
            if (A[i][i] == 0.0) return;
            x[i] = (b[i] - sum) / A[i][i];
        }
    }
}

int main()
{
    double xi = -1.0;
    double xf = 2.0;
    double vi = -1.0;
    double vf = 1.0;
    int n = PARTICION;
    double h = (xf - xi) / (n - 1);

    double A[N][N] = {0};  // Inicializar toda la matriz a cero
    double b[N] = {0};
    double x[N] = {0};

    double R = 1.0 / (h * h);
    double P = -2.0 / (h * h);
    double Q = 1.0 / (h * h);

    // Primer renglón
    A[0][0] = P;
    A[0][1] = Q;
    b[0] = LadoDerecho(xi) - vi * R;

    // Renglones intermedios
    for (int i = 1; i < N - 1; i++) {
        A[i][i - 1] = R;
        A[i][i] = P;
        A[i][i + 1] = Q;
        b[i] = LadoDerecho(xi + h * i);
    }

    // Renglón final (uso de h*(N-1) y coeficiente R para la frontera)
    A[N - 1][N - 2] = R;
    A[N - 1][N - 1] = P;
    b[N - 1] = LadoDerecho(xi + h * (N - 1)) - vf * R;

    // Resolver
    Gauss_Seidel(A, x, b, N, 1000);
    Jacobi(A, x, b, N, 1000);

    FILE *file = fopen("resultados.dat", "w");
    if (!file) {
        perror("Error al abrir resultados.dat");
        return 1;
    }

    for (int i = 0; i < N; i++) {
        fprintf(file, "%f %f\n", xi + i * h, x[i]);
    }
    fclose(file);

    printf("Resultados escritos en 'resultados.dat'\n");
    return 0;
}

/*
Compilar con:
    gcc programa.c -lm -o programa
*/

