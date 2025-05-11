! Este programa resuelve la ecuacion de Poisson bidimensional (laplacian(V)=rho)
! en el dominio rectangular 1 <= x <= 2, 0 <= y <= 2.
! Utiliza el método de relajación Gauss-Seidel con parámetros fijos.
!
! Parametros: Nx=100, Ny=200, eps=1e-7.
! Dominio: Lx=1, Ly=2 (de x=1 a x=2, y=0 a y=2).
! hx = Lx/Nx, hy = Ly/Ny. Asumimos grilla cuadrada hx = hy = h.
! Para Lx=1, Ly=2 y Nx=100, necesitamos Ny=200 para hx=hy.
!
! Termino fuente (rho):
! rho(x,y) = 4 en el interior del dominio.

PROGRAM Poisson2D_ShiftedDomain
  implicit none
  ! Declaración de variables
  real(8),parameter::PI=3.141592653589793d0 ! Valor de pi con doble precisión
  ! Parametros fijos
  integer,parameter::Nx = 200           ! Número de intervalos en X
  integer,parameter::Ny = 200           ! Número de intervalos en Y (Ajustado para grilla cuadrada con Lx=1, Ly=2)
  real(8),parameter::eps = 1.0d-7       ! Criterio de convergencia para la relajación

  real(8),allocatable::phi(:,:),rho(:,:) ! Matriz de potencial (phi) y término fuente (rho=4)
  real(8)::Lx,Ly                      ! Dimensiones del dominio rectangular (tamaño)
  real(8)::xmin, ymin                  ! Coordenadas del origen del dominio
  real(8)::h                           ! Tamaño del paso de la grilla (asumimos hx=hy)
  real(8)::source_rho = 4.0d0          ! Valor del término fuente (constante)

  logical,allocatable::conductor(:,:) ! Matriz booleana para identificar puntos conductores (fronteras)

  ! --- Datos del problema (fijos para este caso específico) ---
  xmin = 0.0d0   ! Coordenada minima en X
  ymin = 0.0d0   ! Coordenada minima en Y
  Lx = 2.0d0 - xmin ! Longitud del dominio en X (2-0 = 2)
  Ly = 2.0d0 - ymin ! Longitud del dominio en Y (2-0 = 2)

  ! Calcular tamaño de paso h (asumiendo grilla cuadrada hx=hy)
  ! Verificamos si Lx/Nx == Ly/Ny
  if (abs(Lx/real(Nx,kind=8) - Ly/real(Ny,kind=8)) > 1.0d-9) then
      write(6,*) "Error: La grilla no es cuadrada con Nx, Ny dados en este dominio."
      write(6,*) "Lx/Nx =", Lx/real(Nx,kind=8), " Ly/Ny =", Ly/real(Ny,kind=8)
      write(6,*) "Para una grilla cuadrada con Lx=1, Ly=2 y Nx=100, Ny deberia ser ", int(Ly/Lx * Nx)
      stop "Detenido debido a tamanio de grilla no cuadrada."
  endif
  h = Lx / real(Nx, kind=8) ! o Ly / real(Ny, kind=8)

  write(6,*)'-------------------------------------------------------'
  write(6,*)' Resolviendo la ecuacion de Poisson bidimensional'
  write(6,"(a,f6.1,a,f6.1,a,f6.1,a,f6.1)")' Dominio: [', xmin, ',', xmin+Lx, '] x [', ymin, ',', ymin+Ly, ']'
  write(6,"(a,i4,' X ',i4)")' Puntos en la grilla: ',Nx+1,Ny+1
  write(6,"(a,f10.5)")' Tamanio de paso h = ', h
  write(6,"(a,f10.7)")' Criterio de convergencia eps = ', eps
  write(6,"(a,f5.1)")' Termino fuente rho = ', source_rho
  write(6,*)' Condiciones de frontera Dirichlet:'
  write(6,"(a)")'  V(1,y) = y^2'
  write(6,"(a)")'  V(2,y) = (y - 2)^2'
  write(6,"(a)")'  V(x,0) = x^2'
  write(6,"(a)")'  V(x,2) = (x - 2)^2'
  write(6,*)'-------------------------------------------------------'

  ! --- Asignar memoria para las matrices (Nx+1 x Ny+1 puntos) ---
  allocate(phi(0:Nx,0:Ny), rho(0:Nx,0:Ny), conductor(0:Nx,0:Ny))

  ! Inicializar matrices
  phi = 0.d0
  conductor = .FALSE.
  ! rho es el término fuente, que es 4 en el interior para este problema
  rho = source_rho

  ! --- Inicializa la red y aplica condiciones de frontera ---
  ! Pasamos xmin, ymin para calcular las coordenadas físicas correctas
  ! Tambien pasamos el valor del termino fuente
  call red_inicial_poisson_shifted(Nx, Ny, h, xmin, ymin, conductor, phi, source_rho) ! Modified call

  ! --- Calcula la ecuación de Poisson iterativamente (Gauss-Seidel) ---
  call Poisson_Solver_Shifted(Nx, Ny, h, eps, conductor, rho, phi) ! Modified call

  ! --- Guarda resultados ---
  ! Pasamos xmin, ymin para guardar las coordenadas físicas correctas
  call guardar_poisson_shifted(Nx, Ny, h, xmin, ymin, phi) ! Modified call

  ! --- Liberar memoria ---
  deallocate(phi, rho, conductor)

END PROGRAM Poisson2D_ShiftedDomain

!****************************************************************
! Subrutina para inicializar la red, aplicar condiciones de frontera
! para un dominio desplazado, y establecer rho (constante = 4).
SUBROUTINE red_inicial_poisson_shifted(Nx,Ny,h,xmin,ymin,conductor,phi,source_rho)
  implicit none
  ! Argumentos de entrada
  integer,INTENT(IN)::Nx,Ny
  real(8),INTENT(IN)::h, xmin, ymin
  real(8),INTENT(IN)::source_rho ! El valor del término fuente

  ! Argumentos de salida e entrada/salida
  real(8),dimension(0:Nx,0:Ny),INTENT(OUT)::phi
  logical,dimension(0:Nx,0:Ny),INTENT(OUT)::conductor
  ! rho ya no se inicializa aqui, se asume inicializado en el main
  ! real(8),dimension(0:Nx,0:Ny),INTENT(OUT)::rho ! Removed

  ! Variables locales
  integer::i,j
  real(8)::x,y                       ! Coordenadas físicas

  ! Inicializa todas las celdas con potencial 0 y no conductoras
  phi = 0.d0
  conductor = .FALSE.
  ! rho = source_rho ! rho es inicializado en el programa principal ahora

  ! --- Ponemos las condiciones de frontera (marcadas como conductores) ---
  ! Los puntos en la frontera son "conductores" en el sentido de que su potencial es fijo.
  conductor(:,0) = .TRUE.  ! Lado inferior (y=ymin)
  conductor(:,Ny) = .TRUE. ! Lado superior (y=ymin+Ly)
  conductor(0,:) = .TRUE.  ! Lado izquierdo (x=xmin)
  conductor(Nx,:) = .TRUE. ! Lado derecho (x=xmin+Lx)

  ! --- Asignar potenciales fijos en las fronteras (usando las nuevas condiciones) ---
  ! Las coordenadas físicas se calculan usando xmin, ymin y h

  ! Frontera inferior (y=ymin=0): V(x,0) = x^2
  do i=0,Nx
     x = xmin + i * h
     phi(i,0) = x**2
  enddo

  ! Frontera izquierda (x=xmin=1): V(1,y) = y^2
  do j=0,Ny
     y = ymin + j * h
     phi(0,j) = y**2
  enddo

  ! Frontera superior (y=ymin+Ly=2): V(x,2) = (x - 2)^2
  do i=0,Nx
     x = xmin + i * h
     phi(i,Ny) = (x - 2.0d0)**2
  enddo

  ! Frontera derecha (x=xmin+Lx=2): V(2,y) = (y - 1)^2
  do j=0,Ny
     y = ymin + j * h
     phi(Nx,j) = (y - 2.0d0)**2
  enddo

  ! NOTA: Verificar la consistencia de las condiciones de frontera en las esquinas:
  ! (xmin, ymin) = (1, 0): V(1,0) = 0^2 = 0  AND 1^2 = 1. -> INCONSISTENT!
  ! (xmin+Lx, ymin) = (2, 0): V(2,0) = (0-1)^2 = 1 AND 2^2 = 4. -> INCONSISTENT!
  ! (xmin, ymin+Ly) = (1, 2): V(1,2) = 2^2 = 4 AND (1-2)^2 = 1. -> INCONSISTENT!
  ! (xmin+Lx, ymin+Ly) = (2, 2): V(2,2) = (2-1)^2 = 1 AND (2-2)^2 = 0. -> INCONSISTENT!
  ! Todas las esquinas tienen condiciones inconsistentes. La solucion numerica
  ! fijará el potencial en las esquinas segun la ultima asignacion en los
  ! bucles de frontera. Esto puede afectar la suavidad de la solucion
  ! cerca de esas esquinas.

  ! --- Distribución del término fuente en el interior ---
  ! La matriz rho debe contener el valor del término fuente en el interior.
  ! En este caso, rho(x,y) = 4 en todo el dominio, ya inicializado en el main.

END SUBROUTINE red_inicial_poisson_shifted

!****************************************************************
! Subrutina para resolver la ecuación de Poisson (rho=4)
! usando el método de relajación Gauss-Seidel.
! Formula de actualizacion usa el promedio de vecinos y el termino fuente.
SUBROUTINE Poisson_Solver_Shifted(Nx,Ny,h,eps,conductor,rho,phi)
  implicit none
  ! Argumentos de entrada
  integer,INTENT(IN)::Nx,Ny
  real(8),INTENT(IN)::h            ! Tamaño del paso de la grilla
  real(8),INTENT(IN)::eps          ! Criterio de convergencia

  ! Argumentos de entrada (pero sus valores no cambian en el interior)
  logical,dimension(0:Nx,0:Ny),INTENT(IN)::conductor
  real(8),dimension(0:Nx,0:Ny),INTENT(IN)::rho ! Termino fuente (constante=4)

  ! Argumentos de entrada/salida (el potencial se actualiza)
  real(8),dimension(0:Nx,0:Ny),INTENT(INOUT)::phi

  ! Variables locales
  integer::i,j,iconteo              ! Contadores
  real(8)::phi_ij                   ! Nuevo valor calculado para phi(i,j)
  real(8)::error                    ! Error máximo en una iteración
  real(8)::dphi                     ! Cambio en el potencial de una celda en una iteración
  integer,parameter::max_iter=1000000 ! Límite máximo de iteraciones

  iconteo = 0 ! Inicializar contador de iteraciones

  ! Bucle principal de iteración
  do while (.TRUE.)
    error = 0.d0 ! Reiniciar el error máximo para esta iteración

    ! Iterar sobre los puntos interiores de la grilla
    do i=1,Nx-1
      do j=1,Ny-1
        ! Cambiamos el potencial solo para los puntos que no son conductores (el interior)
        if(.NOT.conductor(i,j))then
          ! Fórmula de actualización de Gauss-Seidel para la ecuación de Poisson:
          ! V_ij = 0.25 * ( V_i+1,j + V_i-1,j + V_i,j+1 + V_i,j-1 - h^2 * rho_ij )
          phi_ij = 0.25d0 * (phi(i+1,j) + phi(i-1,j) + phi(i,j+1) + phi(i,j-1) - h**2 * rho(i,j))

          ! Calcular el cambio absoluto en el potencial para este punto
          dphi = abs(phi(i,j) - phi_ij)

          ! Actualizar el error máximo si este cambio es mayor
          if(error .lt. dphi) error = dphi

          ! Actualizar el potencial en este punto con el nuevo valor (Gauss-Seidel)
          phi(i,j) = phi_ij
        endif
      enddo
    enddo

    iconteo = iconteo + 1 ! Incrementar el contador de iteraciones

    ! Criterio de salida: si el error máximo es menor que la precisión requerida
    if(error .lt. eps) exit

    ! Límite de iteraciones para evitar bucles infinitos en caso de no convergencia
    if(iconteo > max_iter) then
        write(6,*) 'Advertencia: Máximo número de iteraciones (', max_iter, ') alcanzado antes de la convergencia.'
        write(6,*) 'Error maximo final = ',error
        exit
    end if

  enddo ! Fin del bucle principal de iteración

  write(6,*)' Iteraciones para converger: ',iconteo,' Error maximo final = ',error

END SUBROUTINE Poisson_Solver_Shifted

!****************************************************************
! Subrutina para guardar los resultados numéricos de la solución de Poisson.
! Las coordenadas fisicas se calculan con el desplazamiento (xmin, ymin).
SUBROUTINE guardar_poisson_shifted(Nx,Ny,h,xmin,ymin,phi)
  implicit none
  ! Argumentos de entrada
  integer,INTENT(IN)::Nx,Ny             ! Dimensiones de la grilla (N intervalos)
  real(8),INTENT(IN)::h                 ! Tamaño del paso de la grilla
  real(8),INTENT(IN)::xmin, ymin        ! Coordenadas de inicio del dominio

  ! Argumento de entrada (la matriz de potencial calculada)
  real(8),dimension(0:Nx,0:Ny),INTENT(IN)::phi

  ! Variables locales
  integer::i,j                       ! Contadores
  real(8)::x,y                         ! Coordenadas físicas

  ! Abrir archivo para guardar los datos
  ! unit=1 es un número de unidad de archivo, file="data_poisson_shifted.txt"
  open(unit=1,file="./data_poisson-03-fortran.txt", status="replace") ! status="replace" sobrescribe si existe

  ! Escribir una línea de encabezado en el archivo (opcional, útil para gnuplot)
  write(1, '("# x y Potential(numerical)")')

  ! Iterar sobre todos los puntos de la grilla (incluyendo fronteras)
  do i=0,Nx
    do j=0,Ny
      ! Calcular coordenadas físicas (x, y) con el desplazamiento
      x = xmin + i * h
      y = ymin + j * h

      ! --- Escribir los datos en el archivo ---
      ! Formato: x, y, Potencial Numérico
      write(1,"(2(F10.5,1x),F15.7)")x, y, phi(i,j)
    enddo
    ! Escribir una línea en blanco después de cada fila de la grilla para gnuplot (splot)
    write(1,*)
  enddo

  ! Cerrar el archivo
  close(unit=1)

  write(6,*)' Resultados guardados en ./data_poisson-03-fortran.txt'

END SUBROUTINE guardar_poisson_shifted
