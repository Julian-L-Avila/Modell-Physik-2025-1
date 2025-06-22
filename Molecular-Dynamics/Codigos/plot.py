#!/usr/bin/env python 

import sys,math
from vpython import *

try:
    arch = sys.argv[1]
    s=int(sys.argv[2])
except:
    print("Use: "+sys.argv[0]+" archivo"+" velocidad_movimiento")
    sys.exit(1)


# paredes de la caja 
esp = 1 #espesor 
L=40 #longitud [-L,L] 
s2=2*L-esp # ancho de las paredes 
s3=2*L+esp
pR=box(pos=vector(L,0,0),length=esp,height=s2,width=s3,color=vector(0.7,0.7,0.1)) 
pL=box(pos=vector(-L,0,0),length=esp,height=s2,width=s3,color=vector(0.7,0.7,0.1)) 
pB=box(pos=vector(0,-L,0),length=s3,height=esp,width=s3,color=vector(0.7,0.7,0.1))
pT=box(pos=vector(0, L,0),length=s3,height=esp,width=s3,color=vector(0.7,0.7,0.1))
pBK=box(pos=vector(0,0,-L),length=s2,height=s2,width=esp,color=vector(0.8,0.8,0.2))

# leo archivo
infile=open(arch,'r')
lines=infile.readlines()
infile.close()
ntot=int(lines[0]) #numero de particulas
dat=lines[1].split()
Lc=float(dat[1])
print
print('ntot= ',ntot,'L= ',Lc)
scale=2*L/Lc  #porque Lc es el lado total de la caja: (-Lc/2,Lc/2)
#posiciones leidas y posiciones en la pantalla
vr=[]
for line in lines[2:ntot+2]:
    dat=line.split()
    x=float(dat[1])*scale
    y=float(dat[2])*scale
    z=float(dat[3])*scale
    xx=vector(x,y,z)
    vr.append(xx)

#se crean las esferas 
p=[] 
for i in range(ntot):
    ball=sphere(color=vector(0.4,0.2,0.9),radius=1) 
    ball.pos=vr[i]
    p.append(ball)

frames=int(len(lines)/(ntot+2))-1
print('frames= ',frames+1)

# ciclo sin fin 
while (1==1):
    #ciclo de movimiento de las particulas
    for i in range(frames):
        rate(s) 
        j=i*(ntot+2)
        for ball in p:  
            vr=[]
            for line in lines[j+2:j+2+ntot]:
                dat=line.split()
                x=float(dat[1])*scale
                y=float(dat[2])*scale
                z=float(dat[3])*scale
                xx=vector(x,y,z)
                vr.append(xx)
            for k in range(ntot): 
                p[k].pos=vr[k]




