! Este programa resuelve la ecuación de Laplace bidimensional
! en un dominio rectangular [1,2]×[0,1] con condiciones de frontera no constantes.
! Utiliza el método de relajación Gauss-Seidel con tolerancia fija.

PROGRAM Laplace2D_Rect
  implicit none
  ! Parámetros del dominio
  real(8), parameter :: x_min = 1.0d0, x_max = 2.0d0
  real(8), parameter :: y_min = 0.0d0, y_max = 1.0d0
  integer :: Nx, Ny
  real(8) :: hx, hy
  real(8), parameter :: eps = 1.0d-7  ! Tolerancia fija
  real(8), allocatable :: phi(:,:), rho(:,:)
  logical, allocatable :: conductor(:,:)
  integer :: i, j
  real(8) :: x, y

  ! Leer tamaño de la malla
  write(6,*) 'Número de puntos en X (Nx)?'
  read(5,*) Nx
  write(6,*) 'Número de puntos en Y (Ny)?'
  read(5,*) Ny
  hx = (x_max - x_min) / real(Nx,8)
  hy = (y_max - y_min) / real(Ny,8)
  write(6,'(A,I4,A,I4)') 'Malla: ', Nx, ' x ', Ny
  write(6,'(A,2F10.5)') 'Pasos (hx, hy) = ', hx, hy
  write(6,'(A,ES12.4)') 'Tolerancia eps = ', eps

  ! Asignar memoria
  allocate(phi(0:Nx,0:Ny), rho(0:Nx,0:Ny), conductor(0:Nx,0:Ny))
  phi = 0.d0
  rho = 0.d0
  conductor = .false.

  ! Inicializar contornos
  ! Lado y=0: V(x,0) = 2*log(x)
  do i = 0, Nx
    x = x_min + i*hx
    phi(i,0) = 2.d0 * log(x)
    conductor(i,0) = .true.
  end do
  ! Lado y=1: V(x,1) = 2*log(x) + 4
  do i = 0, Nx
    x = x_min + i*hx
    phi(i,Ny) = 2.d0 * log(x) + 4.d0
    conductor(i,Ny) = .true.
  end do
  ! Lado x=1 (índice 0): V(1,y) = log(y**2 + 1)
  do j = 0, Ny
    y = y_min + j*hy
    phi(0,j) = log(y**2 + 1.d0)
    conductor(0,j) = .true.
  end do
  ! Lado x=2 (índice Nx): V(2,y) = log(y**2 + 4)
  do j = 0, Ny
    y = y_min + j*hy
    phi(Nx,j) = log(y**2 + 4.d0)
    conductor(Nx,j) = .true.
  end do

  ! Resolver Laplace por Gauss-Seidel
  call SolveLaplace(Nx, Ny, hx, hy, eps, conductor, phi)

  ! Guardar datos y comparar con solución analítica V=log(x**2 + y**2)
  call SaveResults(Nx, Ny, hx, hy, x_min, y_min, phi)

  deallocate(phi, rho, conductor)
END PROGRAM Laplace2D_Rect


! Subrutina Gauss-Seidel
SUBROUTINE SolveLaplace(Nx, Ny, hx, hy, eps, conductor, phi)
  implicit none
  integer, intent(in) :: Nx, Ny
  real(8), intent(in) :: hx, hy, eps
  logical, dimension(0:Nx,0:Ny), intent(in) :: conductor
  real(8), dimension(0:Nx,0:Ny), intent(inout) :: phi
  integer :: i, j, iter, max_iter
  real(8) :: error, dphi, phi_new

  max_iter = 100000
  iter = 0
  do
    error = 0.d0
    do i = 1, Nx-1
      do j = 1, Ny-1
        if (.not. conductor(i,j)) then
          ! Gauss-Seidel en malla no uniforme (hx, hy)
          phi_new = ((phi(i+1,j) + phi(i-1,j))/hx**2 + (phi(i,j+1) + phi(i,j-1))/hy**2) / (2.d0*(1.d0/hx**2 + 1.d0/hy**2))
          dphi = abs(phi(i,j) - phi_new)
          if (dphi > error) error = dphi
          phi(i,j) = phi_new
        end if
      end do
    end do
    iter = iter + 1
    if (error < eps .or. iter > max_iter) exit
  end do
  write(6,'(A,I8,A,ES12.4)') 'Iter: ', iter, '  Error final = ', error
END SUBROUTINE SolveLaplace


! Subrutina para guardar datos y solución analítica
SUBROUTINE SaveResults(Nx, Ny, hx, hy, x0, y0, phi)
  implicit none
  integer, intent(in) :: Nx, Ny
  real(8), intent(in) :: hx, hy, x0, y0
  real(8), dimension(0:Nx,0:Ny), intent(in) :: phi
  integer :: i, j
  real(8) :: x, y, van
  open(unit=10, file='data_laplace_rect.txt', status='replace')
  write(10,'(A)') '# x y V_numeric V_analytic'
  do i = 0, Nx
    x = x0 + i*hx
    do j = 0, Ny
      y = y0 + j*hy
      van = log(x**2 + y**2)
      write(10,'(2F12.6,2F15.8)') x, y, phi(i,j), van
    end do
    write(10,*)
  end do
  close(10)
  write(6,*) 'Datos guardados en data_laplace_rect.txt'
END SUBROUTINE SaveResults
