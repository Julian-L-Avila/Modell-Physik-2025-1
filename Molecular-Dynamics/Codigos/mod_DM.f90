MODULE modDM
  REAL(8),PARAMETER::PI=3.14159265358979323846d0

  ! Npar = numero total de particulas
  ! T    = temperatura del sistema
  integer,save:: Npar,Ndm,neq,div,iseed,reini
  real(8),save:: dt,T,Etot,Epot,Ecin,Rcut,Ecut
  real(8),save:: rho,L,Rcut2,dt2,Tc,L2
  real(8),save,allocatable::vr(:,:),vv(:,:),vF(:,:),r_old(:,:)
  character(len=30),save::fsave,festruc
  character(len=24),save::ftemp2
  character(len=2),save,allocatable::el(:)
  
  ! Nfree = numero de grados de libertad
  integer,save:: Nfree

  INTERFACE
     DOUBLE PRECISION FUNCTION ran2(idum)
       INTEGER(selected_int_kind(9)), INTENT(INOUT) :: idum
     END FUNCTION ran2
  END INTERFACE

END MODULE modDM





