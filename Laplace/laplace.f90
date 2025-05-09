! Este programa resuelve la ecuación de Laplace bidimensional
! en un dominio cuadrado de 1x1 con condiciones de frontera Dirichlet.
! Utiliza el método de relajación Gauss-Seidel.

PROGRAM Laplace2D
  implicit none
  ! Declaración de variables
  real(8),parameter::PI=3.141592653589793d0 ! Valor de pi con doble precisión
  integer::Nx,Ny                      ! Número de puntos en la grilla en direcciones X e Y
  real(8),allocatable::phi(:,:),rho(:,:) ! Matriz de potencial (phi) y densidad de carga (rho)
  real(8)::Vx1,Vx2,Vy1,Vy2             ! Potenciales en las fronteras (y=0, y=1, x=0, x=1)
  real(8)::Q                          ! Carga total (0 para Laplace)
  real(8)::eps                         ! Criterio de convergencia para la relajación
  real(8)::Lx,Ly                      ! Dimensiones del dominio rectangular
  real(8)::h                           ! Tamaño del paso de la grilla
  real(8)::area                       ! Área del dominio
  real(8)::rho0                       ! Densidad de carga promedio (no usada en Laplace)
  logical,allocatable::conductor(:,:) ! Matriz booleana para identificar puntos conductores (fronteras)

  ! --- Datos del problema (fijos para este caso específico) ---
  Lx = 1.0d0   ! Longitud del dominio en X (1 metro)
  Ly = 1.0d0   ! Longitud del dominio en Y (1 metro)
  Vx1 = 0.0d0  ! Potencial en la frontera inferior (y=0) (0 Voltios)
  Vx2 = 200.0d0! Potencial en la frontera superior (y=1) (200 Voltios)
  Vy1 = 0.0d0  ! Potencial en la frontera izquierda (x=0) (0 Voltios)
  Vy2 = 0.0d0  ! Potencial en la frontera derecha (x=1) (0 Voltios)
  Q = 0.0d0    ! Carga total (0 para la ecuación de Laplace)

  write(6,*)'-------------------------------------------------------'
  write(6,*)' Resolviendo la ecuacion de Laplace bidimensional'
  write(6,*)' Dominio: Cuadrado 1m x 1m'
  write(6,*)' Condiciones de frontera:'
  write(6,"(a,f6.1,a)")'  V = ', Vx1, ' V en y = 0'
  write(6,"(a,f6.1,a)")'  V = ', Vx2, ' V en y = 1'
  write(6,"(a,f6.1,a)")'  V = ', Vy1, ' V en x = 0'
  write(6,"(a,f6.1,a)")'  V = ', Vy2, ' V en x = 1'
  write(6,*)'-------------------------------------------------------'

  ! --- Información de la grilla ---
  write(6,*)' Para un cuadrado 1x1, el número de puntos en X e Y es el mismo.'
  write(6,*)' Número de puntos en la grilla (por eje)? (Ej: 50 para una grilla de 50x50)'
  read(5,*)Nx
  Ny = Nx ! Para una grilla cuadrada, Nx = Ny

  ! Calcular tamaño de paso h
  h = Lx / real(Nx, kind=8)

  write(6,"(a,i4,' X ',i4)")' Puntos en la grilla: ',Nx,Ny
  write(6,"(a,f10.5)")' Tamaño de paso h = ', h

  ! --- Asignar memoria para las matrices ---
  allocate(phi(0:Nx,0:Ny), rho(0:Nx,0:Ny), conductor(0:Nx,0:Ny))

  ! Inicializar matrices a cero
  phi = 0.d0
  rho = 0.d0 ! rho es cero en el interior para la ecuación de Laplace

  ! rho0 no es relevante para la ecuación de Laplace (Q=0)
  area = Lx * Ly
  rho0 = Q / area ! Esto será 0 ya que Q=0

  ! --- Precision para la convergencia ---
  write(6,*)' Precision requerida para la convergencia (eps)? (Ej: 1e-6)'
  read(5,*)eps

  ! --- Inicializa la red y aplica condiciones de frontera ---
  ! Pasamos rho0, aunque no se usa para establecer rho en el interior en Laplace.
  call red_inicial(Vx1, Vx2, Vy1, Vy2, Nx, Ny, rho0, conductor, rho, phi)

  ! --- Calcula la ecuación de Laplace iterativamente (Gauss-Seidel) ---
  ! La subrutina Poisson resuelve la ecuación de Poisson.
  ! Como rho es cero, resuelve la ecuación de Laplace.
  call Poisson(Nx, Ny, eps, conductor, rho, phi)

  ! --- Guarda resultados ---
  ! Pasamos Lx y Ly para el cálculo de la solución analítica.
  call guardar(Nx, Ny, h, phi, Lx, Ly, Vx2) ! Pasamos Vx2 (el potencial V0)

  ! --- Liberar memoria ---
  deallocate(phi, rho, conductor)

END PROGRAM Laplace2D

!****************************************************************
! Subrutina para inicializar la red, aplicar condiciones de frontera
! y establecer la densidad de carga (cero para Laplace).
SUBROUTINE red_inicial(Vx1,Vx2,Vy1,Vy2,Nx,Ny,rho0_unused,conductor,rho,phi)
  implicit none
  ! Argumentos de entrada
  integer,INTENT(IN)::Nx,Ny
  real(8),INTENT(IN)::Vx1,Vx2,Vy1,Vy2
  real(8),INTENT(IN)::rho0_unused ! No usado para establecer rho en el interior en Laplace

  ! Argumentos de salida e entrada/salida
  real(8),dimension(0:Nx,0:Ny),INTENT(OUT)::rho,phi
  logical,dimension(0:Nx,0:Ny),INTENT(OUT)::conductor

  ! Variables locales
  integer::i,j

  ! Inicializa todas las celdas con potencial 0 y no conductoras, rho a 0
  phi = 0.d0
  conductor = .FALSE.
  rho = 0.d0 ! La densidad de carga es cero en el interior para la ecuación de Laplace

  ! --- Ponemos las condiciones de frontera (marcadas como conductores) ---
  ! Los puntos en la frontera son "conductores" en el sentido de que su potencial es fijo.
  conductor(:,0) = .TRUE.  ! Lado inferior del rectangulo (y=0)
  conductor(:,Ny) = .TRUE. ! Lado superior del rectangulo (y=Ly)
  conductor(0,:) = .TRUE.  ! Lado izquierdo del rectangulo (x=0)
  conductor(Nx,:) = .TRUE. ! Lado derecho del rectangulo (x=Lx)

  ! --- Asignar potenciales fijos en las fronteras ---
  phi(:,0) = Vx1 ! Potencial en y=0
  phi(:,Ny) = Vx2 ! Potencial en y=Ly
  phi(0,:) = Vy1 ! Potencial en x=0
  phi(Nx,:) = Vy2 ! Potencial en x=Lx

  ! --- Distribución de carga en el interior ---
  ! Para la ecuación de Laplace, la densidad de carga en el interior es cero.
  ! La inicialización de rho a 0.d0 al inicio de la subrutina ya se encarga de esto.
  ! El siguiente bucle comentado se usaría si hubiera una distribución de carga no nula.
  ! do i=1,Nx-1
  !    do j=1,Ny-1
  !       rho(i,j) = ... ! Aquí se establecería la densidad de carga si no fuera cero
  !    enddo
  ! enddo

END SUBROUTINE red_inicial

!****************************************************************
! Subrutina para resolver la ecuación de Poisson (o Laplace si rho=0)
! usando el método de relajación Gauss-Seidel.
SUBROUTINE Poisson(Nx,Ny,eps,conductor,rho,phi)
  implicit none
  ! Argumentos de entrada
  integer,INTENT(IN)::Nx,Ny
  real(8),INTENT(IN)::eps          ! Criterio de convergencia

  ! Argumentos de entrada (pero sus valores no cambian en el interior)
  logical,dimension(0:Nx,0:Ny),INTENT(IN)::conductor
  real(8),dimension(0:Nx,0:Ny),INTENT(INOUT)::rho ! rho es entrada para la ecuación

  ! Argumentos de entrada/salida (el potencial se actualiza)
  real(8),dimension(0:Nx,0:Ny),INTENT(INOUT)::phi

  ! Variables locales
  integer::i,j,iconteo              ! Contadores
  real(8)::phi_ij                   ! Nuevo valor calculado para phi(i,j)
  real(8)::error                    ! Error máximo en una iteración
  real(8)::dphi                     ! Cambio en el potencial de una celda en una iteración
  integer,parameter::max_iter=100000 ! Límite máximo de iteraciones

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
          ! phi(i,j)_nuevo = 0.25 * (phi(i+1,j) + phi(i-1,j) + phi(i,j+1) + phi(i,j-1) + h^2/epsilon * rho(i,j))
          ! Para la ecuación de Laplace, rho(i,j) = 0, y se asume h=1 para la discretización
          ! simplificada que lleva a la media de los vecinos.
          ! Si se usara la forma completa discretizada con h, sería:
          ! phi_ij = 0.25d0 * (phi(i+1,j) + phi(i-1,j) + phi(i,j+1) + phi(i,j-1)) - (h**2/4.0d0) * rho(i,j)
          ! La fórmula en el código original asume rho incluye el factor h^2/epsilon y otros constantes,
          ! o que la ecuación de Poisson se discretizó de forma que lleve a esta forma simple
          ! cuando rho se define de cierta manera. Dado que rho(i,j) es 0 para Laplace,
          ! el término rho(i,j) desaparece, dejando la fórmula para Laplace:
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
        exit
    end if

  enddo ! Fin del bucle principal de iteración

  write(6,*)' Iteraciones para converger: ',iconteo,' Error maximo final = ',error

END SUBROUTINE Poisson

!****************************************************************
! Subrutina para guardar los resultados numéricos y la solución analítica
! en un archivo para su posterior visualización o análisis.
SUBROUTINE guardar(Nx,Ny,h,phi, Lx, Ly, V0_boundary)
  implicit none
  ! Argumentos de entrada
  integer,INTENT(IN)::Nx,Ny             ! Dimensiones de la grilla
  real(8),INTENT(IN)::h                 ! Tamaño del paso de la grilla
  real(8),INTENT(IN)::Lx,Ly             ! Dimensiones físicas del dominio
  real(8),INTENT(IN)::V0_boundary       ! Potencial en la frontera superior (y=Ly), para la analítica

  ! Argumento de entrada (la matriz de potencial calculada)
  real(8),dimension(0:Nx,0:Ny),INTENT(IN)::phi

  ! Variables locales
  integer::i,j,n                       ! Contadores
  real(8)::x,y                         ! Coordenadas físicas
  real(8)::vxy_analytical              ! Valor del potencial según la solución analítica
  real(8)::pi_val                      ! Valor de pi
  integer,parameter::num_terms = 200   ! Número de términos en la serie de Fourier (debe ser par para ir hasta n=num_terms-1)

  pi_val = 3.14159265358979323846d0 ! Valor de pi con mayor precisión

  ! Abrir archivo para guardar los datos
  ! unit=1 es un número de unidad de archivo, file="data_laplace.txt" es el nombre del archivo
  open(unit=1,file="data_laplace.txt", status="replace") ! status="replace" sobrescribe si existe

  ! Escribir una línea de encabezado en el archivo (opcional, útil para gnuplot)
  ! Se corrigió la falta del paréntesis inicial en el formato.
  write(1, '("# x y Potential(numerical) Potential(analytical)")')

  ! Iterar sobre todos los puntos de la grilla (incluyendo fronteras)
  do i=0,Nx
    do j=0,Ny
      ! Calcular coordenadas físicas (x, y)
      x = i * h
      y = j * h

      ! --- Calcular la solución analítica ---
      ! Solución para V(x,0)=0, V(x,Ly)=V0, V(0,y)=0, V(Lx,y)=0 en un cuadrado Lx x Ly
      ! V(x,y) = Sum_{n odd} (4*V0 / (n*pi)) * (sinh(n*pi*y/Lx) / sinh(n*pi*Ly/Lx)) * sin(n*pi*x/Lx)
      vxy_analytical = 0.d0

      ! Suma de la serie de Fourier (usando términos impares)
      do n = 1, num_terms - 1, 2 ! Iterar solo sobre n = 1, 3, 5, ... hasta num_terms-1
         ! Se dividió la línea larga usando el caracter de continuación '&'
         vxy_analytical = vxy_analytical + ( sinh(n * pi_val * y / Lx) / &
                          sinh(n * pi_val * Ly / Lx) ) * &
                          sin(n * pi_val * x / Lx) / real(n, kind=8)
      end do
      ! Multiplicar por el factor constante de la serie
      vxy_analytical = vxy_analytical * (4.0d0 * V0_boundary / pi_val)

      ! --- Escribir los datos en el archivo ---
      ! Formato: x, y, Potencial Numérico, Potencial Analítico
      write(1,"(2(F10.5,1x),2(F15.7,1x))")x, y, phi(i,j), vxy_analytical
    enddo
    ! Escribir una línea en blanco después de cada fila de la grilla para gnuplot (splot)
    write(1,*)
  enddo

  ! Cerrar el archivo
  close(unit=1)

  write(6,*)' Resultados guardados en data_laplace.txt'

END SUBROUTINE guardar

