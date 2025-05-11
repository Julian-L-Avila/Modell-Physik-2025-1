PROGRAM Poisson2D_ShiftedDomain
  implicit none
  ! Declaración de variables
  real(8),parameter::PI=3.141592653589793d0 ! Valor de pi con doble precisión
  ! Parametros fijos
  integer,parameter::Nx = 100        ! Número de intervalos en X
  integer,parameter::Ny = 50        ! Número de intervalos en Y (para Lx=1, Ly=1 => Nx=Ny para hx=hy)
  real(8),parameter::eps = 1.0d-7     ! Criterio de convergencia para la relajación

  real(8),allocatable::phi(:,:),rho(:,:) ! Matriz de potencial (phi) y término fuente (rho)
  real(8)::Lx,Ly                     ! Dimensiones del dominio rectangular (tamaño)
  real(8)::xmin, ymin                  ! Coordenadas del origen del dominio
  real(8)::h                         ! Tamaño del paso de la grilla (asumimos hx=hy)

  logical,allocatable::is_boundary(:,:) ! Matriz booleana para identificar puntos de frontera

  ! --- Datos del problema ---
  xmin = 0.0d0  ! Coordenada minima en X
  ymin = 0.0d0  ! Coordenada minima en Y
  Lx = 2.0d0    ! Longitud del dominio en X (xmax = xmin + Lx = 2.0)
  Ly = 1.0d0    ! Longitud del dominio en Y (ymax = ymin + Ly = 2.0)

  ! Calcular tamaño de paso h (asumiendo grilla cuadrada hx=hy)
  if (abs(Lx/real(Nx,kind=8) - Ly/real(Ny,kind=8)) > 1.0d-9) then
     write(6,*) "Error: La grilla no es cuadrada con Nx, Ny, Lx, Ly dados."
     write(6,*) "Lx/Nx =", Lx/real(Nx,kind=8), " Ly/Ny =", Ly/real(Ny,kind=8)
     write(6,*) "Para una grilla cuadrada con Lx=", Lx, ", Ly=", Ly, " y Nx=", Nx
     write(6,*) "Ny deberia ser ", int(Ly/Lx * Nx)
     stop "Detenido debido a tamanio de grilla no cuadrada."
  endif
  h = Lx / real(Nx, kind=8)

  write(6,*)'-------------------------------------------------------'
  write(6,*)' Resolviendo la ecuacion de Poisson bidimensional'
  write(6,"(a,f6.2,a,f6.2,a,f6.2,a,f6.2)")' Dominio: [', xmin, ',', xmin+Lx, '] x [', ymin, ',', ymin+Ly, ']'
  write(6,"(a,i4,' X ',i4)")' Puntos en la grilla: ',Nx+1,Ny+1
  write(6,"(a,f10.5)")' Tamanio de paso h = ', h
  write(6,"(a,f10.7)")' Criterio de convergencia eps = ', eps
  write(6,*)'-------------------------------------------------------'

  allocate(phi(0:Nx,0:Ny), rho(0:Nx,0:Ny), is_boundary(0:Nx,0:Ny))

  phi = 0.d0
  is_boundary = .FALSE.
  rho = 0.d0

  call initialize_grid_and_boundaries(Nx, Ny, h, xmin, ymin, Lx, Ly, is_boundary, phi, rho)

  call Poisson_Solver(Nx, Ny, h, eps, is_boundary, rho, phi)

  call save_results(Nx, Ny, h, xmin, ymin, phi)

  deallocate(phi, rho, is_boundary)

END PROGRAM Poisson2D_ShiftedDomain

!****************************************************************
! Subrutina para inicializar la red, aplicar condiciones de frontera
! y establecer el término fuente rho(x,y).
!****************************************************************
SUBROUTINE initialize_grid_and_boundaries(Nx,Ny,h,xmin,ymin,Lx,Ly,is_boundary,phi,rho)
  implicit none
  integer,INTENT(IN)::Nx,Ny
  real(8),INTENT(IN)::h, xmin, ymin, Lx, Ly
  real(8),dimension(0:Nx,0:Ny),INTENT(OUT)::phi,rho
  logical,dimension(0:Nx,0:Ny),INTENT(OUT)::is_boundary
  integer::i,j
  real(8)::x,y
  real(8)::xmax, ymax

  xmax = xmin + Lx
  ymax = ymin + Ly

  phi = 0.d0
  rho = 0.d0
  is_boundary = .FALSE.

  is_boundary(:,0) = .TRUE.   ! Lado inferior (y=ymin)
  is_boundary(:,Ny) = .TRUE.  ! Lado superior (y=ymax)
  is_boundary(0,:) = .TRUE.   ! Lado izquierdo (x=xmin)
  is_boundary(Nx,:) = .TRUE.  ! Lado derecho (x=xmax)

  do i=0,Nx
     x = xmin + i * h
     if (x <= 0.0d0) then
        phi(i,0) = 0.0d0 ! Or handle error; with xmin=1, x is always > 0
     else
        phi(i,0) = 1.0d0
     endif
  enddo

  do j=0,Ny
     y = ymin + j * h
     if (y <= 0.0d0) then
        phi(0,j) = 0.0d0 ! Or handle error; with ymin=1, y is always > 0
     else
        phi(0,j) = 1.0d0
     endif
  enddo

  do i=0,Nx
     x = xmin + i * h
     if (x <= 0.0d0 .or. (4.0d0 * x) <= 0.0d0) then
        phi(i,Ny) = 0.0d0 ! Or handle error
     else
        phi(i,Ny) = exp(x)
     endif
  enddo

  do j=0,Ny
     y = ymin + j * h
     if (y <= 0.0d0 .or. (2.0d0 * y) <= 0.0d0) then
        phi(Nx,j) = 0.0d0 ! Or handle error
     else
        phi(Nx,j) = exp(2 * y)
     endif
  enddo

  ! Establecer el término fuente rho(x,y) = x/y + y/x para puntos interiores
  ! Rho en las fronteras no se usa en la actualización de Gauss-Seidel para phi,
  ! pero definimos rho(i,j) para todos los puntos.
  do i=0,Nx
    x = xmin + i * h
    do j=0,Ny
      y = ymin + j * h
      ! Con xmin=1, ymin=1, x e y siempre serán >= 1, so y no será cero.
      ! No es necesario un chequeo de y == 0.0d0 aquí con el dominio actual.
      if (is_boundary(i,j)) then
          rho(i,j) = 0.0d0 ! O el valor de x/y+y/x, aunque no se use. Cero es más limpio.
      else
          if (abs(y) < 1.0d-12 .or. abs(x) < 1.0d-12) then ! General safety for division
             rho(i,j) = 1.0d+20 ! Large number to indicate issue if it somehow occurs
             write(*,*) "Warning: x or y is near zero for rho calc at interior: ", x, y
          else
             rho(i,j) = (x*x + y*y) * exp(x * y)
          endif
      endif
    enddo
  enddo

END SUBROUTINE initialize_grid_and_boundaries

!****************************************************************
! Subrutina para resolver la ecuación de Poisson
! usando el método de relajación Gauss-Seidel.
!****************************************************************
SUBROUTINE Poisson_Solver(Nx,Ny,h,eps,is_boundary,rho,phi)
  implicit none
  integer,INTENT(IN)::Nx,Ny
  real(8),INTENT(IN)::h
  real(8),INTENT(IN)::eps
  logical,dimension(0:Nx,0:Ny),INTENT(IN)::is_boundary
  real(8),dimension(0:Nx,0:Ny),INTENT(IN)::rho
  real(8),dimension(0:Nx,0:Ny),INTENT(INOUT)::phi
  integer::i,j,iter_count
  real(8)::phi_new_ij, max_error, dphi
  integer,parameter::max_iter=1000000
  ! Opcional: Parámetro de relajación para SOR (Successive Over-Relaxation)
  ! real(8), parameter :: omega = 1.8d0 ! Típicamente 1 < omega < 2

  iter_count = 0

  do while (.TRUE.)
     max_error = 0.d0

     do j=1,Ny-1 ! Iterar sobre y (columnas)
        do i=1,Nx-1 ! Iterar sobre x (filas)
           if(.NOT.is_boundary(i,j))then
              phi_new_ij = 0.25d0 * (phi(i+1,j) + phi(i-1,j) + &
                                     phi(i,j+1) + phi(i,j-1) - h**2 * rho(i,j))

              ! --- Opcional: Successive Over-Relaxation (SOR) ---
              ! phi_new_ij = (1.0d0 - omega) * phi(i,j) + omega * phi_new_ij
              ! --- Fin SOR ---

              dphi = abs(phi(i,j) - phi_new_ij)
              if(max_error < dphi) max_error = dphi
              phi(i,j) = phi_new_ij
           endif
        enddo
     enddo

     iter_count = iter_count + 1

     if(max_error < eps) exit
     if(iter_count > max_iter) then
        write(6,*) 'Advertencia: Maximo numero de iteraciones (', max_iter, ') alcanzado.'
        write(6,*) 'Error maximo final = ',max_error
        exit
     end if
     ! if (mod(iter_count, 500) == 0) then ! Descomentar para ver progreso
     !    write(6,"(a,i8,a,e12.5)") "Iter:", iter_count, " Max Error:", max_error
     ! endif
  enddo

  write(6,*)' Iteraciones para converger: ',iter_count,' Error maximo final = ',max_error

END SUBROUTINE Poisson_Solver

!****************************************************************
! Subrutina para guardar los resultados numéricos de la solución de Poisson.
!****************************************************************
SUBROUTINE save_results(Nx,Ny,h,xmin,ymin,phi)
  implicit none
  integer,INTENT(IN)::Nx,Ny
  real(8),INTENT(IN)::h, xmin, ymin
  real(8),dimension(0:Nx,0:Ny),INTENT(IN)::phi
  integer::i,j
  real(8)::x,y
  integer, parameter :: out_unit = 11 ! Usar una unidad de archivo > 10

  open(unit=out_unit, file="data_poisson_shifted.txt", status="replace")

  write(out_unit, '("# x y Potential(numerical)")')

  do j=0,Ny ! Gnuplot prefiere que las 'y' varíen más lentamente para splot pm3d
     do i=0,Nx
        x = xmin + i * h
        y = ymin + j * h
        write(out_unit,"(2(F12.6,1x),F17.9)")x, y, phi(i,j)
     enddo
     write(out_unit,*) ! Línea en blanco para Gnuplot splot
  enddo

  close(unit=out_unit)
  write(6,*)' Resultados guardados en data_poisson_shifted.txt'

END SUBROUTINE save_results
