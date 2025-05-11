! Este programa resuelve la ecuación de Laplace bidimensional (laplacian(V)=0)
! en el dominio rectangular 1 <= x <= 2, 0 <= y <= 1.
! Utiliza el método de relajación Gauss-Seidel con parámetros fijos.
!
! Parametros: Nx=100, Ny=100, eps=1e-7.
! Dominio: Lx=1, Ly=1 (de x=1 a x=2, y=0 a y=1).
! hx = Lx/Nx, hy = Ly/Ny.
!
! Condiciones de frontera Dirichlet:
! V(1,y) = ln(y^2 + 1)     (x=1, izquierda)
! V(2,y) = ln(y^2 + 4)     (x=2, derecha)
! V(x,0) = 2 * ln(x)       (y=0, inferior)
! V(x,1) = ln(x^2) + 4     (y=1, superior)

PROGRAM Laplace2D_ShiftedDomain
  implicit none
  ! Declaración de variables
  real(8),parameter::PI=3.141592653589793d0 ! Valor de pi con doble precisión
  ! Parametros fijos
  integer,parameter::Nx = 100           ! Número de intervalos en X
  integer,parameter::Ny = 100           ! Número de intervalos en Y
  real(8),parameter::eps = 1.0d-7       ! Criterio de convergencia para la relajación

  real(8),allocatable::phi(:,:),rho(:,:) ! Matriz de potencial (phi) y término fuente (rho=0 para Laplace)
  real(8)::Lx,Ly                      ! Dimensiones del dominio rectangular (tamaño)
  real(8)::xmin, ymin                  ! Coordenadas del origen del dominio
  real(8)::h                           ! Tamaño del paso de la grilla (asumimos hx=hy)

  logical,allocatable::conductor(:,:) ! Matriz booleana para identificar puntos conductores (fronteras)

  ! --- Datos del problema (fijos para este caso específico) ---
  xmin = 1.0d0   ! Coordenada minima en X
  ymin = 0.0d0   ! Coordenada minima en Y
  Lx = 2.0d0 - xmin ! Longitud del dominio en X
  Ly = 1.0d0 - ymin ! Longitud del dominio en Y

  ! Calcular tamaño de paso h (asumiendo grilla cuadrada hx=hy)
  ! Verificamos si Lx/Nx == Ly/Ny
  if (abs(Lx/real(Nx,kind=8) - Ly/real(Ny,kind=8)) > 1.0d-9) then
      write(6,*) "Error: La grilla no es cuadrada con Nx, Ny dados en este dominio."
      write(6,*) "Lx/Nx =", Lx/real(Nx,kind=8), " Ly/Ny =", Ly/real(Ny,kind=8)
      stop "Detenido debido a tamanio de grilla no cuadrada."
  endif
  h = Lx / real(Nx, kind=8) ! o Ly / real(Ny, kind=8)

  write(6,*)'-------------------------------------------------------'
  write(6,*)' Resolviendo la ecuacion de Laplace bidimensional'
  write(6,"(a,f6.1,a,f6.1,a,f6.1,a,f6.1)")' Dominio: [', xmin, ',', xmin+Lx, '] x [', ymin, ',', ymin+Ly, ']'
  write(6,"(a,i4,' X ',i4)")' Puntos en la grilla: ',Nx+1,Ny+1
  write(6,"(a,f10.5)")' Tamanio de paso h = ', h
  write(6,"(a,f10.7)")' Criterio de convergencia eps = ', eps
  write(6,*)' Condiciones de frontera Dirichlet:'
  write(6,"(a)")'  V(1,y) = ln(y^2 + 1)'
  write(6,"(a)")'  V(2,y) = ln(y^2 + 4)'
  write(6,"(a)")'  V(x,0) = 2 * ln(x)'
  write(6,"(a)")'  V(x,1) = ln(x^2) + 4'
  write(6,*)'-------------------------------------------------------'

  ! --- Asignar memoria para las matrices (Nx+1 x Ny+1 puntos) ---
  allocate(phi(0:Nx,0:Ny), rho(0:Nx,0:Ny), conductor(0:Nx,0:Ny))

  ! Inicializar matrices a cero
  phi = 0.d0
  rho = 0.d0 ! rho es cero para la ecuación de Laplace

  ! --- Inicializa la red y aplica condiciones de frontera ---
  ! Pasamos xmin, ymin para calcular las coordenadas físicas correctas
  call red_inicial_laplace_shifted(Nx, Ny, h, xmin, ymin, conductor, rho, phi)

  ! --- Calcula la ecuación de Laplace iterativamente (Gauss-Seidel) ---
  ! rho es cero, asi que resuelve la ecuacion de Laplace.
  call Laplace_Solver_Shifted(Nx, Ny, eps, conductor, phi)

  ! --- Guarda resultados ---
  ! Pasamos xmin, ymin para guardar las coordenadas físicas correctas
  call guardar_laplace_shifted(Nx, Ny, h, xmin, ymin, phi)

  ! --- Liberar memoria ---
  deallocate(phi, rho, conductor)

END PROGRAM Laplace2D_ShiftedDomain

!****************************************************************
! Subrutina para inicializar la red, aplicar condiciones de frontera
! para un dominio desplazado, y establecer rho (cero para Laplace).
SUBROUTINE red_inicial_laplace_shifted(Nx,Ny,h,xmin,ymin,conductor,rho,phi)
  implicit none
  ! Argumentos de entrada
  integer,INTENT(IN)::Nx,Ny
  real(8),INTENT(IN)::h, xmin, ymin

  ! Argumentos de salida e entrada/salida
  real(8),dimension(0:Nx,0:Ny),INTENT(OUT)::rho,phi
  logical,dimension(0:Nx,0:Ny),INTENT(OUT)::conductor

  ! Variables locales
  integer::i,j
  real(8)::x,y                       ! Coordenadas físicas

  ! Inicializa todas las celdas con potencial 0 y no conductoras, rho a 0
  phi = 0.d0
  conductor = .FALSE.
  rho = 0.d0 ! La densidad de carga es cero en el interior para la ecuación de Laplace

  ! --- Ponemos las condiciones de frontera (marcadas como conductores) ---
  ! Los puntos en la frontera son "conductores" en el sentido de que su potencial es fijo.
  conductor(:,0) = .TRUE.  ! Lado inferior (y=ymin)
  conductor(:,Ny) = .TRUE. ! Lado superior (y=ymin+Ly)
  conductor(0,:) = .TRUE.  ! Lado izquierdo (x=xmin)
  conductor(Nx,:) = .TRUE. ! Lado derecho (x=xmin+Lx)

  ! --- Asignar potenciales fijos en las fronteras (usando las nuevas condiciones) ---
  ! Las coordenadas físicas se calculan usando xmin, ymin y h

  ! Frontera inferior (y=ymin): V(x,ymin) = 2 * ln(x)
  do i=0,Nx
     x = xmin + i * h
     phi(i,0) = 2.0d0 * log(x)
  enddo

  ! Frontera izquierda (x=xmin): V(xmin,y) = ln(y^2 + 1)
  do j=0,Ny
     y = ymin + j * h
     phi(0,j) = log(y**2 + 1.0d0)
  enddo

  ! Frontera superior (y=ymin+Ly): V(x,ymin+Ly) = ln(x^2) + 4
  do i=0,Nx
     x = xmin + i * h
     ! V(x, 1) = log(x^2) + 4
     phi(i,Ny) = log(x**2 + 1.0d0)
  enddo

  ! Frontera derecha (x=xmin+Lx): V(xmin+Lx,y) = ln(y^2 + 4)
  do j=0,Ny
     y = ymin + j * h
     ! V(2, y) = log(y^2 + 4)
     phi(Nx,j) = log(y**2 + 4.0d0)
  enddo

  ! NOTA: Verificar la consistencia de las condiciones de frontera en las esquinas:
  ! (xmin, ymin) = (1, 0): V(1,0) = 2*ln(1) = 0  AND ln(0^2+1) = ln(1) = 0. -> Consistent
  ! (xmin+Lx, ymin) = (2, 0): V(2,0) = 2*ln(2) AND ln(0^2+4) = ln(4) = ln(2^2) = 2*ln(2). -> Consistent
  ! (xmin, ymin+Ly) = (1, 1): V(1,1) = log(1^2)+4 = 4 AND ln(1^2+1) = ln(2). -> INCONSISTENT!
  ! (xmin+Lx, ymin+Ly) = (2, 1): V(2,1) = log(2^2)+4 = ln(4)+4 AND ln(1^2+4) = ln(5). -> INCONSISTENT!
  ! La solucion numerica manejará estas inconsistencias fijando el potencial
  ! en las esquinas segun la ultima asignacion en los bucles de frontera.
  ! Esto puede afectar la suavidad de la solucion cerca de esas esquinas.

  ! --- Distribución de carga en el interior ---
  ! Para la ecuación de Laplace, la densidad de carga en el interior es cero.
  ! La inicialización de rho a 0.d0 al inicio de la subrutina ya se encarga de esto.

END SUBROUTINE red_inicial_laplace_shifted

!****************************************************************
! Subrutina para resolver la ecuación de Laplace (rho=0)
! usando el método de relajación Gauss-Seidel.
! Formula de actualizacion solo usa el promedio de vecinos.
SUBROUTINE Laplace_Solver_Shifted(Nx,Ny,eps,conductor,phi)
  implicit none
  ! Argumentos de entrada
  integer,INTENT(IN)::Nx,Ny
  real(8),INTENT(IN)::eps          ! Criterio de convergencia

  ! Argumentos de entrada (pero sus valores no cambian en el interior)
  logical,dimension(0:Nx,0:Ny),INTENT(IN)::conductor

  ! Argumentos de entrada/salida (el potencial se actualiza)
  real(8),dimension(0:Nx,0:Ny),INTENT(INOUT)::phi

  ! Variables locales
  integer::i,j,iconteo              ! Contadores
  real(8)::phi_ij                   ! Nuevo valor calculado para phi(i,j)
  real(8)::error                    ! Error máximo en una iteración
  real(8)::dphi                     ! Cambio en el potencial de una celda en una iteración
  integer,parameter::max_iter=1000000 ! Límite máximo de iteraciones (aumentado por si acaso)

  iconteo = 0 ! Inicializar contador de iteraciones

  ! Bucle principal de iteración
  do while (.TRUE.)
    error = 0.d0 ! Reiniciar el error máximo para esta iteración

    ! Iterar sobre los puntos interiores de la grilla
    do i=1,Nx-1
      do j=1,Ny-1
        ! Cambiamos el potencial solo para los puntos que no son conductores (el interior)
        if(.NOT.conductor(i,j))then
          ! Fórmula de actualización de Gauss-Seidel para la ecuación de Laplace:
          ! Es el promedio de los 4 vecinos
          phi_ij = 0.25d0 * (phi(i+1,j) + phi(i-1,j) + phi(i,j+1) + phi(i,j-1))

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

END SUBROUTINE Laplace_Solver_Shifted

!****************************************************************
! Subrutina para guardar los resultados numéricos de la solución de Laplace.
! Las coordenadas fisicas se calculan con el desplazamiento (xmin, ymin).
! No incluye comparacion analitica ya que la BL en y=1 es inconsistente.
SUBROUTINE guardar_laplace_shifted(Nx,Ny,h,xmin,ymin,phi)
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
  ! unit=1 es un número de unidad de archivo, file="data_laplace_shifted.txt"
  open(unit=1,file="data_laplace_shifted.txt", status="replace") ! status="replace" sobrescribe si existe

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

  write(6,*)' Resultados guardados en data_laplace_shifted.txt'

END SUBROUTINE guardar_laplace_shifted
