!PROGRAMA REALIZADO POR JOSE MEJIA LOPEZ
!ultima actualizacion: 15/04/2020
!aqui se especifica el estado inicial del sistema
SUBROUTINE inicio
  USE modDM
  implicit none
  integer:: i,j,n,nx,ny,nz,ib,nn
  real(8)::c(3),a,b(4,3),xmax(3),xmin(3),vcm(3)
  real(8)::r
  character(len=10)::ch1
  character(len=30)::archE,archC
  LOGICAL::w
  
  !Leo coordenadas desde coordenas.ini si existe
  INQUIRE(FILE='coordenadas.ini',EXIST=w)
  if(w)then
     open(1,file='coordenadas.ini',status='old')
     read(1,*)n; read(1,*)
     if(n.ne.Npar)then
        write(6,*)'el archivo coordenadas.ini existe pero Npar no coincide'
        stop
     end if
     do i=1,Npar
        read(1,*)el(i),vr(i,:),vv(i,:)
     end do
     close(1)
     write(6,*)'se leyo coordenadas desde el archivo coordenadas.ini'
  else
     a=(4.d0/rho)**(1.d0/3.d0); n=ceiling(L/a)
     write(6,*)'L= ',L,'   a= ',a
     j=4*n**3-Npar; i=0; nn=0
     b=0.d0
     b(2,1)=0.5d0*a; b(2,2)=0.5d0*a
     b(3,1)=0.5d0*a; b(3,3)=0.5d0*a
     b(4,2)=0.5d0*a; b(4,3)=0.5d0*a
     do nx=0,n-1
        c(1)=nx*a
        do ny=0,n-1
           c(2)=ny*a
           do nz=0,n-1
              c(3)=nz*a           
              do ib=1,4
                 if(i.lt.j)then
                    r=ran2(iseed)
                    if(r.gt.0.5)then
                       nn=nn+1; vr(nn,:)=c(:)+b(ib,:)
                    end if
                 else
                    nn=nn+1; vr(nn,:)=c(:)+b(ib,:) 
                 end if
              end do
           end do
        end do
     end do
     el='Fe'
     write(6,*)'Las coordenadas se generaron aleatoriamente en una fcc'
     write(6,*)'nn= ',nn
     !pongo velocidades aleatoriamente entre -0.5 y 0.5
     do i=1,Npar
        vv(i,1)=ran2(iseed)-0.5d0
        vv(i,2)=ran2(iseed)-0.5d0
        vv(i,3)=ran2(iseed)-0.5d0
     end do
  end if

  !traslada el centro a cero y las velocidades con respecto al cm
  xmax(:)=vr(1,:); xmin=xmax
  do i=1,Npar
     where(vr(i,:).gt.xmax)xmax(:)=vr(i,:)
     where(vr(i,:).lt.xmin)xmin(:)=vr(i,:)
  end do
  c=0.5d0*(xmax+xmin)
  vcm=0.d0; Ecin=0.d0
  do i=1,Npar
     vr(i,:)=vr(i,:)-c(:)
     vcm(:)=vcm(:)+vv(i,:)
  end do
  vcm=vcm/Npar
  do i=1,Npar
     vv(i,:)=(vv(i,:)-vcm(:)) !sistema de referencia del centro de masa
     Ecin=Ecin+dot_product(vv(i,:),vv(i,:))
  end do
  Tc=Ecin/Nfree
  vv=vv*sqrt(T/Tc) !para tener la temperatura correcta
  Ecin=0.5d0*Ecin/Npar !energia cinetica inicial
  r_old=vr-vv*dt   !posiciones anteriores

  !guarda configuracion inicial en formato xyz
  open(9,file='conf_inicial.xyz',status='unknown')
  write(9,*)Npar
  write(9,*)
  do i=1,Npar
     write(9,"('Fe',3(1x,F16.9))")vr(i,:)
  end do
  close(9)

  !guarda configuracion inicial en formato xyz
  open(9,file='conf_inicial.vasp',status='unknown')
  write(9,"(a)")'comentario'
  write(9,"(F16.12)")1.d0
  write(9,"(3(1x,F16.12))")L,0.d0,0.d0
  write(9,"(3(1x,F16.12))")0.d0,L,0.d0
  write(9,"(3(1x,F16.12))")0.d0,0.d0,L
  write(9,"(a)")'Fe'
  write(9,"(I6)")Npar
  write(9,"(a)")'Cartesian'
  do i=1,Npar
     write(9,"(3(1x,F19.12))")vr(i,:)
  end do
  close(9)

  !abre archivos para escritura
  archE="E_"//trim(fsave); archC='C_'//trim(fsave)
  open(2,file=archC,status='unknown')
  open(3,file=archE,status='unknown')
  write(3,"('# stept',7x,'Etot',7x,'Epot',7x,'Temper')")

  !guardo valores iniciales
  call fuerzas
  write(2,*)Npar; write(2,*)'L=',L,'nt= ',0
  do i=1,Npar
     write(2,"(a2,3(1x,F11.6))")el(i),vr(i,:)
  enddo
  write(3,"(I6,3(1x,F11.6))")0,Epot+0.5d0*Nfree*T/Npar,Epot,T

END SUBROUTINE inicio

