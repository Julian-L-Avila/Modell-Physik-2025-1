!PROGRAMA REALIZADO POR JOSE MEJIA LOPEZ
!ultima actualizacion: 16/04/2020
!algoritmo de Verlet ensenble microcanonico
SUBROUTINE integracion(nt)
  USE modDM
  implicit none
  integer,INTENT(IN)::nt
  integer::i,j
  real(8)::xx(3),vcm(3),c(3)

  vcm=0.d0; Ecin=0.d0
  do i =1,Npar
     xx(:)=2.d0*vr(i,:)-r_old(i,:)+vF(i,:)*dt2

     !condiciones de borde periodica
     where(abs(xx(:)).gt.L2)&
          xx(:)=xx(:)-L*anint(xx(:)/L) 
     c(:)=xx(:)-r_old(i,:)
     where(abs(c(:)).gt.L2)&
          c(:)=c(:)-L*anint(c(:)/L)
     !
     
     vv(i,:)=0.5d0*c(:)/dt
     vcm(:)=vcm(:)+vv(i,:)
     Ecin=Ecin+dot_product(vv(i,:),vv(i,:))

     r_old(i,:)=vr(i,:)
     vr(i,:)=xx(:)

  end do
  Tc=Ecin/Nfree        !calculo la temperatura instantanea
  Ecin=0.5d0*Ecin/Npar !energia cinetica por particula
  Etot=Ecin+Epot       !energia total por particula
  
END SUBROUTINE integracion

