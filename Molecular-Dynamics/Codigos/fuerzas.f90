!PROGRAMA REALIZADO POR JOSE MEJIA LOPEZ
!ultima actualizacion: 15/04/2020
!calcula las fuerzas entre particulas y la energia potencial del sistema,
SUBROUTINE fuerzas
  USE modDM
  implicit none
  integer::i,j
  real(8)::r(3),r2,r_2,r_6,ff

  Epot=0.d0; vF=0.d0
  do i=1,Npar-1
     do j=i+1,Npar
        r(:)=vr(i,:)-vr(j,:)
        r=r-L*Anint(r/L) !condiciones de borde periodica
        r2=dot_product(r,r)
        if(r2.lt.Rcut2)then
           r_2=1.d0/r2; r_6=r_2**3
           ff=r_2*r_6*(r_6-0.5d0)
           vF(i,:)=vF(i,:)+ff*r(:)
           vF(j,:)=vF(j,:)-ff*r(:)
           Epot=Epot+r_6*(r_6-1.d0)
        endif
     end do
  end do
  Epot=(Epot*4.d0-Ecut)/Npar  !energia potencial por particula
  vF=vF*48.d0

END SUBROUTINE fuerzas


