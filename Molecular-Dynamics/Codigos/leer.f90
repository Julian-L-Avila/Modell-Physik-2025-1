!PROGRAMA REALIZADO POR JOSE MEJIA LOPEZ
!ultima actualizacion: 15/04/2020
!lee los datos de input.dat y los pone en el lenguaje que se utilizara 
!en la simulacion.
SUBROUTINE leer
  USE modDM
  IMPLICIT NONE
  integer::ii(3),ens,i,j
  character(len=20)::ch1

  open(1,FILE='input.dat',STATUS='old',ERR=80)
  !archivo donde se guarda
  read(1,*); read(1,*)fsave
  !parametros fisicos
  read(1,*); read(1,*)ch1,Npar
  read(1,*)ch1,rho
  read(1,*)ch1,T
  !parametros dinamicos
  read(1,*); read(1,*)ch1,dt
  read(1,*)ch1,Ndm
  read(1,*)ch1,neq
  read(1,*)ch1,div
  !reiniciacion
  read(1,*); read(1,*)ch1,reini
  close(1)

  write(6,22)trim(fsave),Npar,rho,dt,Ndm,T,div,reini
  
22 format('# output ',a,2x,'     Npar=',i7,'   rho=',F5.2/,&  
        '# dt = ',F10.7,4x,'Ndm= ',I10,4x,'T= ',F14.6,/,&  
        '    div= ',i10,'   reini= ',i4)

  !localiza memoria
  allocate(vr(Npar,3),el(Npar),vv(Npar,3),vF(Npar,3),r_old(Npar,3))

  !grados de libertad y condiciones iniciales
  Nfree=3*Npar-3  !momento lineal constante
  L=(Npar/rho)**(1.d0/3.d0)
  L2=0.5d0*L

  !parametros 
  Rcut=0.5d0*L   !radio de corte
  Rcut2=Rcut**2
  Ecut=2.d0*(1.d0/Rcut**12-1.d0/Rcut**6)*Npar*(Npar-1) 
  write(6,"(a,F14.6)")"Ecut/N = ",Ecut/Npar
  dt2=dt**2
  iseed=-384957835
    
  RETURN
80 write(6,*)
  write(6,*)'No existe el archivo  input.dat '
  write(6,*)
  STOP
  
END SUBROUTINE leer
