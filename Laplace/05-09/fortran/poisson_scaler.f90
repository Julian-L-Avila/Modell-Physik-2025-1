! Este programa resuelve la ecuación de Poisson bidimensional
! en un dominio cuadrado de Lx*Ly con condiciones de frontera Dirichlet.
! Utiliza el método de relajación Gauss-Seidel.
!
! Se usa un tamanio de paso h constante, por lo que el tamanio del dominio
! Lx, Ly cambia con el numero de intervalos N (Lx = N*h).
!
! Ecuacion: laplacian(V) = (x^2 + y^2) * exp(xy)
! Fronteras: V(0,y)=1, V(x,0)=1, V(Lx,y)=exp(2y), V(x,Ly)=exp(x)
! (Note: Boundary conditions now depend on the changing domain size Lx, Ly)

PROGRAM Poisson2D_Scaler
  implicit none
  ! Declaración de variables
  real(8),parameter::PI=3.141592653589793d0 ! Valor de pi con doble precisión
  real(8),parameter::h = 0.01d0          ! <-- CONSTANTE: Tamanio del paso de la grilla
  integer::Nx,Ny                      ! Número de intervalos en la grilla en direcciones X e Y (= N)
  real(8),allocatable::phi(:,:),rho(:,:) ! Matriz de potencial (phi) y término fuente discretizado (rho)
  real(8)::eps                         ! Criterio de convergencia para la relajación
  real(8)::Lx,Ly                      ! Dimensiones del dominio rectangular (cambian con N)
  ! real(8)::area                      ! No directamente relevante para este Poisson

  logical,allocatable::conductor(:,:) ! Matriz booleana para identificar puntos conductores (fronteras)

  ! --- Información de la grilla (N intervalos = N+1 puntos) ---
  write(6,*)'-------------------------------------------------------'
  write(6,*)' Resolviendo la ecuacion de Poisson bidimensional'
  write(6,*)' Usando un tamanio de paso h = ', h
  write(6,*)' Ingrese el numero de intervalos en la grilla (N):'
  read(5,*)Nx
  Ny = Nx ! Dominio cuadrado N x N intervalos = (N+1) x (N+1) puntos

  ! Calcular el tamaño del dominio basado en N y h constante
  Lx = real(Nx, kind=8) * h
  Ly = real(Ny, kind=8) * h

  write(6,"(a,i4,' X ',i4)")' N intervalos en la grilla: ',Nx,Ny
  write(6,"(a,f10.5)")' Tamanio del dominio Lx = ', Lx
  write(6,"(a,f10.5)")' Tamanio del dominio Ly = ', Ly
  write(6,*)' Condiciones de frontera Dirichlet:'
  write(6,"(a)")'  V = 1        en x = 0'
  write(6,"(a)")'  V = 1        en y = 0'
  write(6,"(a,f10.5,a)")'  V = exp(2y)  en x = Lx = ', Lx
  write(6,"(a,f10.5,a)")'  V = exp(x)   en y = Ly = ', Ly
  write(6,*)'-------------------------------------------------------'


  ! --- Asignar memoria para las matrices (N+1 x N+1 puntos) ---
  allocate(phi(0:Nx,0:Ny), rho(0:Nx,0:Ny), conductor(0:Nx,0:Ny))

  ! Inicializar matrices a cero
  phi = 0.d0
  rho = 0.d0 ! rho se usará para el término fuente (x^2+y^2)exp(xy) en el interior

  ! --- Precision para la convergencia ---
  write(6,*)' Precision requerida para la convergencia (eps)? (Ej: 1e-6)'
  read(5,*)eps

  ! --- Inicializa la red, aplica condiciones de frontera y establece el término fuente ---
  ! Pasamos Lx y Ly ya que las condiciones de frontera dependen del tamaño del dominio
  call red_inicial_poisson_scaler(Nx, Ny, h, Lx, Ly, conductor, rho, phi)

  ! --- Calcula la ecuación de Poisson iterativamente (Gauss-Seidel) ---
  ! Ahora rho contiene el término fuente, asi que resuelve la ecuacion de Poisson completa.
  call Poisson_Solver_hconst(Nx, Ny, h, eps, conductor, rho, phi)

  ! --- Guarda resultados ---
  ! No necesitamos pasar Lx, Ly ya que la escritura solo usa indices y h
  call guardar_poisson_scaler(Nx, Ny, h, phi)

  ! --- Liberar memoria ---
  deallocate(phi, rho, conductor)

END PROGRAM Poisson2D_Scaler

!****************************************************************
! Subrutina para inicializar la red, aplicar condiciones de frontera
! variables (que dependen del tam. del dominio) y establecer el término fuente.
SUBROUTINE red_inicial_poisson_scaler(Nx,Ny,h,Lx,Ly,conductor,rho,phi)
  implicit none
  ! Argumentos de entrada
  integer,INTENT(IN)::Nx,Ny
  real(8),INTENT(IN)::h, Lx, Ly ! h es constante, Lx, Ly dependen de N

  ! Argumentos de salida e entrada/salida
  real(8),dimension(0:Nx,0:Ny),INTENT(OUT)::rho,phi
  logical,dimension(0:Nx,0:Ny),INTENT(OUT)::conductor

  ! Variables locales
  integer::i,j
  real(8)::x,y                       ! Coordenadas físicas

  ! Inicializa todas las celdas con potencial 0 y no conductoras, rho a 0
  phi = 0.d0
  conductor = .FALSE.
  rho = 0.d0 ! Inicializamos rho a 0, luego calcularemos el término fuente en el interior

  ! --- Ponemos las condiciones de frontera (marcadas como conductores) ---
  ! Los puntos en la frontera son "conductores" en el sentido de que su potencial es fijo.
  conductor(:,0) = .TRUE.  ! Lado inferior (y=0)
  conductor(:,Ny) = .TRUE. ! Lado superior (y=Ly)
  conductor(0,:) = .TRUE.  ! Lado izquierdo (x=0)
  conductor(Nx,:) = .TRUE. ! Lado derecho (x=Lx)

  ! --- Asignar potenciales fijos en las fronteras (usando las nuevas condiciones) ---
  ! Frontera inferior (y=0): V(x,0) = 1
  do i=0,Nx
     phi(i,0) = 1.0d0
  enddo

  ! Frontera izquierda (x=0): V(0,y) = 1
  do j=0,Ny
     phi(0,j) = 1.0d0
  enddo

  ! Frontera superior (y=Ly): V(x,Ly) = exp(x)
  do i=0,Nx
     x = i * h
     ! La condicion es V(x, Ly) = exp(x)
     phi(i,Ny) = exp(x)
  enddo

  ! Frontera derecha (x=Lx): V(Lx,y) = exp(2y)
  do j=0,Ny
     y = j * h
     ! La condicion es V(Lx, y) = exp(2y)
     phi(Nx,j) = exp(2.0d0 * y)
  enddo

  ! NOTA: La consistencia de las condiciones de frontera en las esquinas
  ! depende de la funcion de frontera en el punto (Lx, Ly).
  ! V(Lx, Ly) deberia ser exp(Lx) Y exp(2*Ly). Como Lx=Ly=N*h,
  ! V(N*h, N*h) deberia ser exp(N*h) Y exp(2*N*h). Esto solo es consistente
  ! si exp(N*h) = exp(2*N*h), lo cual solo ocurre si N*h=0, o si el problema
  ! no requiere consistencia exacta en la esquina (que es comun en problemas numericos).
  ! El codigo actual asigna V(Lx,Ly) con exp(Lx) y exp(2*Ly) sucesivamente,
  ! quedando con el ultimo valor asignado.

  ! --- Establecer el término fuente (x^2 + y^2) * exp(xy) en el interior ---
  ! rho(i,j) en la fórmula discretizada corresponde a f(x_i, y_j)
  do i=1,Nx-1
     do j=1,Ny-1
        x = i * h
        y = j * h
        rho(i,j) = (x**2 + y**2) * exp(x * y)
     enddo
  enddo

END SUBROUTINE red_inicial_poisson_scaler

!****************************************************************
! Subrutina para resolver la ecuación de Poisson (o Laplace si rho=0)
! usando el método de relajación Gauss-Seidel.
! Usa h constante.
SUBROUTINE Poisson_Solver_hconst(Nx,Ny,h,eps,conductor,rho,phi)
  implicit none
  ! Argumentos de entrada
  integer,INTENT(IN)::Nx,Ny
  real(8),INTENT(IN)::h            ! Tamaño del paso (constante)
  real(8),INTENT(IN)::eps          ! Criterio de convergencia

  ! Argumentos de entrada (sus valores no cambian en el interior)
  logical,dimension(0:Nx,0:Ny),INTENT(IN)::conductor
  real(8),dimension(0:Nx,0:Ny),INTENT(IN)::rho ! rho es entrada (contiene el término fuente)

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
          ! V_i,j = 0.25 * Sum(Vecinos) - 0.25 * h^2 * f_i,j
          ! En este código, rho(i,j) = f_i,j
          phi_ij = 0.25d0 * (phi(i+1,j) + phi(i-1,j) + phi(i,j+1) + phi(i,j-1)) &
                 - 0.25d0 * (h**2) * rho(i,j)

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

END SUBROUTINE Poisson_Solver_hconst

!****************************************************************
! Subrutina para guardar los resultados numéricos de la solución de Poisson.
! La comparación con la solución analítica se ha removido.
SUBROUTINE guardar_poisson_scaler(Nx,Ny,h,phi)
  implicit none
  ! Argumentos de entrada
  integer,INTENT(IN)::Nx,Ny             ! Dimensiones de la grilla (N intervalos)
  real(8),INTENT(IN)::h                 ! Tamaño del paso de la grilla (constante)

  ! Argumento de entrada (la matriz de potencial calculada)
  real(8),dimension(0:Nx,0:Ny),INTENT(IN)::phi

  ! Variables locales
  integer::i,j                       ! Contadores
  real(8)::x,y                         ! Coordenadas físicas

  ! Abrir archivo para guardar los datos
  ! unit=1 es un número de unidad de archivo, file="data_poisson.txt" es el nombre del archivo
  ! Cambiamos el nombre para cada N
  character(len=30) :: filename
  write(filename, '(a,i5.5,a)') 'data_poisson_N', Nx, '.txt'

  open(unit=1,file=filename, status="replace") ! status="replace" sobrescribe si existe

  ! Escribir una línea de encabezado en el archivo (opcional, útil para gnuplot)
  write(1, '("# x y Potential(numerical)")')

  ! Iterar sobre todos los puntos de la grilla (incluyendo fronteras)
  do i=0,Nx
    do j=0,Ny
      ! Calcular coordenadas físicas (x, y)
      x = i * h
      y = j * h

      ! --- Escribir los datos en el archivo ---
      ! Formato: x, y, Potencial Numérico
      write(1,"(2(F10.5,1x),F15.7)")x, y, phi(i,j)
    enddo
    ! Escribir una línea en blanco después de cada fila de la grilla para gnuplot (splot)
    write(1,*)
  enddo

  ! Cerrar el archivo
  close(unit=1)

  write(6,"(a,a)")' Resultados guardados en ', trim(filename)

END SUBROUTINE guardar_poisson_scaler
