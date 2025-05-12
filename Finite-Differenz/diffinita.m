% Ejemplo de una funcion para resolver la ecuacion diferencial parcial en 1D
% Uxx = -Pi*Pi*cos(Pi*x)
% xi <= U <= xf
% U(xi) = vi y U(xf) = vf

function diffinita(N_interior)
    xi = -1;          % Inicio de dominio
    xf = 2;           % Fin de dominio
    vi = -1;          % Valor en la frontera xi
    vf = 1;           % Valor en la frontera xf
    N = N_interior;   % Nodos interiores
    n = N + 2;        % Número total de nodos
    h = (xf - xi) / (n - 1);  % Incremento en la malla
    A = zeros(N, N);     % Matriz A
    b = zeros(N, 1);     % Vector b
    x = zeros(N, 1);     % Vector de solución inicial

    R = 1 / (h^2);
    P = -2 / (h^2);
    Q = 1 / (h^2);

    % Primer renglón de la matriz A y vector b
    A(1, 1) = P;
    if N > 1
        A(1, 2) = Q;
    end
    b(1) = LadoDerecho(xi + h) - vi * R; % Ajuste del punto para el lado derecho

    % Renglones intermedios de la matriz A y vector b
    for i = 2:N - 1
        A(i, i - 1) = R;
        A(i, i) = P;
        A(i, i + 1) = Q;
        b(i) = LadoDerecho(xi + h * i); % Ajuste del punto para el lado derecho
    end

    % Renglón final de la matriz A y vector b
    if N > 1
        A(N, N - 1) = R;
    end
    A(N, N) = P;
    b(N) = LadoDerecho(xi + h * N) - vf * Q; % Ajuste del punto para el lado derecho

    % Resuelve el sistema lineal Ax = b usando Gauss-Seidel
    x = gaussSeidel(A, x, b, N, 10000);

    % --- Guardar resultados ---
    filename = sprintf("resultado_%d.dat", N);
    fp = fopen(filename, "w");
    if (fp == -1)
        fprintf(stderr, "Error: No se pudo abrir el archivo %s para escritura.\n", filename);
        return;
    end

    fprintf(fp, "%10.5f %10.5f\n", xi, vi);
    for i = 1:N
        fprintf(fp, "%10.5f %10.5f\n", xi + h * i, x(i));
    end
    fprintf(fp, "%10.5f %10.5f\n", xf, vf);
    fclose(fp);


endfunction

% Lado derecho de la ecuación
function y = LadoDerecho(x)
    y = -pi^2 * cos(pi * x);
endfunction

% Solución analítica a la ecuación (no se usa para la medición de tiempos)
function y = SolucionAnalitica(x)
    y = cos(pi * x);
endfunction

% Función Gauss-Seidel
function x = gaussSeidel(A_in, x_in, b_in, n, iter_max)
    x = x_in;
    for m = 1:iter_max
        for i = 1:n
            sum_val = 0;
            for j = 1:n
                if i ~= j
                    sum_val = sum_val + A_in(i, j) * x(j);
                end
            end
            x(i) = (b_in(i) - sum_val) / A_in(i, i);
        end
    end
endfunction

% La llamada a diffinita con un valor fijo se elimina o comenta
% diffinita(30);
