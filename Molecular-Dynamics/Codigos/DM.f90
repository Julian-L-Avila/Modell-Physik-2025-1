!PROGRAMA REALIZADO POR JOSE MEJIA LOPEZ
!ultima actualizacion: 15/04/2020
! Dinamica Molecular con potenciales de Lennard-Jones
PROGRAM DM
  USE modDM
  implicit none
  integer::i,j,nt,np
  real(8)::cc,time1,time2,pEtot,pEpot,pEcin,pTc
  character(len=30)::archE,archC
  
  CALL CPU_TIME(time1)

  ! lee datos desde input.dat
  call leer

  ! parametros iniciales
  call inicio

  ! Integra ecuaciones de movimiento

  !equilibracion
  do nt=1,neq
     call fuerzas
     call integracion(nt)
     call guardar(nt)
  end do

  !promedios
  pEtot=0.d0; pEpot=0.d0; pEcin=0.d0; pTc=0.d0
  do nt=neq+1,Ndm
     call fuerzas
     call integracion(nt)
     call guardar(nt)
     pEtot=pEtot+Etot; pEpot=pEpot+Epot
     pEcin=pEcin+Ecin; pTc=pTc+Tc
  end do
  np=Ndm-neq
  pEtot=pEtot/np; pEpot=pEpot/np
  pEcin=pEcin/np; pTc=pTc/np

  write(6,"(/,'promedios:',4x,'Etot',8x,'Epot',6x,'Ecin',9x,'T')")
  write(6,"(10x,4(1x,F10.5),/)")pEtot,pEpot,pEcin,pTc

  close(2); close(3)
  
  CALL CPU_TIME(time2)
  write (6,*) "tiempo de corrida:",(time2-time1)/60, " min"
  
END PROGRAM DM

!*********************************
SUBROUTINE guardar(nt)
  USE modDM
  implicit none
  integer,INTENT(IN)::nt
  integer::i
  real(8)::cc,time1,time2
  
  !guardo configuraciones
  if(mod(nt,div).eq.0) then
     write(2,*)Npar; write(2,*)'nt= ',nt
     do i=1,Npar
        write(2,"(a2,3(1x,F11.6))")el(i),vr(i,:)
     enddo
  endif

  !guardo observables
  write(3,"(I6,3(1x,F11.6))")nt,Etot,Epot,Tc
  
END SUBROUTINE guardar

