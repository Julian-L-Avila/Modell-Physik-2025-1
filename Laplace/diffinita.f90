! diffinita.f90 modificado para recibir N desde línea de comandos
program fdm1d
    implicit none
    integer :: i, N, nn, ios
    real*8, allocatable :: A(:,:), b(:), x(:)
    real*8 :: xi, xf, vi, vf, h, R, P, Q, y
    character(len=32) :: arg
    
    ! Leer argumento N (nodos interiores)
    call get_command_argument(1, arg, status=ios)
    if (ios /= 0) then
        print *, "Error: debe pasar N (nodos interiores) como argumento."
        stop 1
    end if
    read(arg, *, iostat=ios) N
    if (ios /= 0 .or. N < 1) then
        print *, "Error: N debe ser entero positivo."
        stop 1
    end if
    
    nn = N + 2    ! incluir puntos de frontera
    
    ! Parámetros del problema
    xi = -1.0d0
    xf =  2.0d0
    vi = -1.0d0
    vf =  1.0d0
    h  = (xf - xi) / (nn - 1)
    
    ! Reservar memoria
    allocate(A(N,N), b(N), x(N))
    A = 0.0d0
    b = 0.0d0
    x = 0.0d0
    
    R =  1.0d0 / (h*h)
    P = -2.0d0 / (h*h)
    Q =  1.0d0 / (h*h)
    
    ! Primer renglón
    A(1,1) = P
    if (N > 1) A(1,2) = Q
    call ladoDerecho(xi + h, y)
    b(1) = y - vi*R
    
    ! Renglones intermedios
    do i = 2, N-1
        A(i,i-1) = R
        A(i,i)   = P
        A(i,i+1) = Q
        call ladoDerecho(xi + i*h, y)
        b(i) = y
    end do
    
    ! Último renglón
    if (N > 1) A(N,N-1) = R
    A(N,N) = P
    call ladoDerecho(xi + N*h, y)
    b(N) = y - vf*Q
    
    ! Resolver sistema por Gauss-Seidel
    call gaussSeidel(A, x, b, N, 10000)
    
    ! Guardar resultados
    open(unit=10, file="resultado_"//trim(adjustl(arg))//".dat", status="replace")
    write(10, '(F10.5,1X,F10.5)') xi, vi
    do i = 1, N
        write(10, '(F10.5,1X,F10.5)') xi + i*h, x(i)
    end do
    write(10, '(F10.5,1X,F10.5)') xf, vf
    close(10)
    
end program fdm1d

subroutine ladoDerecho(x, y)
    implicit none
    real*8, intent(in)  :: x
    real*8, intent(out) :: y
    real*8, parameter :: pi = 3.14159265358979323846d0
    y = -pi*pi * cos(pi*x)
end subroutine ladoDerecho

subroutine gaussSeidel(a, x, b, nx, iter)
    implicit none
    integer, intent(in)    :: nx, iter
    real*8, intent(in)     :: a(nx,nx), b(nx)
    real*8, intent(inout)  :: x(nx)
    integer :: i, j, m
    real*8 :: sum
    
    do m = 1, iter
        do i = 1, nx
            sum = 0.0d0
            do j = 1, nx
                if (i /= j) sum = sum + a(j,i)*x(j)
            end do
            x(i) = (b(i) - sum)/a(i,i)
        end do
    end do
end subroutine gaussSeidel