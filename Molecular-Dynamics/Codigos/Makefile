#============================================
FC      = gfortran
DBG     = 
#DBG    = -pg
#FFLAGS  = -O -C -traceback $(DBG)
FFLAGS  = -O3  $(DBG)
#FFLAGS  = $(DBG)
PROGRAM = xdm

LDR     = f90
LDFLAGS =
 
SRCS    = mod_DM.f90 DM.f90 leer.f90 inicio.f90 fuerzas.f90 integracion.f90 ran2.f90
OBJS    = $(SRCS:.f90=.o)
 
.SUFFIXES: .o .f90

.f90.o:
		$(FC) -c $(FFLAGS) $<
 

$(PROGRAM): $(OBJS) 
		$(FC) $(OBJS) $(FFLAGS) -o $@
 
clean:
		rm *.o *.mod
#================================================
