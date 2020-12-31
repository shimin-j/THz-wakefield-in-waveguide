# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 10:06:42 2019

@author: Shimin
"""

import numpy as np
import cmath
from math import cos, sin, sqrt
import scipy.constants as sc
import scipy.integrate as si
#from numba import jit



#%% The simplied diffraction results

def diffrac_unit(f,R,cosphi,phi,slice1):
    factor = 1j/2/sc.c*f*(1+cosphi)
    U = cmath.exp(-1j*2*sc.pi*f/sc.c*R)/R*factor*cos(phi)*slice1
    return U

################################################################
    ##数值积分不能对复数积分

def plain2plain(phi,f,rm,rn,deltaz,delta,slice1,factor1):
    delta = 0
    R = sqrt(rm**2+rn**2+(deltaz+delta)**2-2*rm*rn*cos(phi))
    cosphi = (deltaz+delta)/R
    unit = diffrac_unit(f,R,cosphi,phi,slice1)
    if factor1 == 1:
        U = unit.real
    else:
        U = unit.imag
    return U

def sphere2sphere(phi,f,thetam,thetan,rho,delta,slice2,factor1):
    R = sqrt(2*rho**2*(1+cos(thetam)*cos(thetan)-sin(thetam)*sin(thetan)*cos(phi))+(delta-rho)**2+2*(delta-rho)*rho*(cos(thetam)+cos(thetan)))
    cosphi = (rho*(cos(thetam)+cos(thetan))+(delta-rho))/R  ##如果说角度超过60度，就不再适合，会在cosphi中出现复数
    unit = diffrac_unit(f,R,cosphi,phi,slice2)
    if factor1 == 1:
        U = unit.real
    else:
        U = unit.imag
    return U

def plain2sphere(phi,f,rm,thetan,rho,delta,slice3,factor1):
    R = sqrt(rm**2+rho**2+delta**2+2*delta*rho*cos(thetan)-2*rho*rm*sin(thetan)*cos(phi))
    cosphi = (rho*cos(thetan)+delta)/R
    unit = diffrac_unit(f,R,cosphi,phi,slice3)
    if factor1 == 1:
        U = unit.real
    else:
        U = unit.imag
    return U

def sphere2plain(phi,f,thetan,rm,rho,delta,slice3,factor1):
    R = sqrt(rm**2+rho**2+delta**2+2*delta*rho*cos(thetan)-2*rho*rm*sin(thetan)*cos(phi))
    cosphi = (rho*cos(thetan)+delta)/R
    unit = diffrac_unit(f,R,cosphi,phi,slice3)
    if factor1 == 1:
        U = unit.real
    else:
        U = unit.imag
    return U

##################################################################

def matrix_diff(func1,f,x11,x22,L,delta):
    '''
    function type, frequncy, first position distribution, next distribtion, distance, error
    '''
    N = np.size(x11)
    M = np.size(x22)
    T = np.zeros((N,M))+1j*np.zeros((N,M))
    ###adjust the type of the function
    if func1 == 'p2s':
        func = plain2sphere
    elif func1 == 'p2p':
        func = plain2plain
    elif func1 == 's2s':
        func = sphere2sphere
    elif func1 == 's2p':
        func = sphere2plain
    else:
        print('function have wrong')
    ### creat the matrix
    for x in range(N):
        for y in range(M):
            ### it is only functional in the first plain
            if func1 == 's2p' or func1 == 's2s':
                slicex1 = L**2*sin(x11[x])*x11[1]
            elif func1 == 'p2s' or func1 == 'p2p':
                slicex1 = x11[x]*x11[1]  
            else:
                slicex1 = 0
            xyreal = si.quad(func,0,sc.pi,args = (f,x11[x],x22[y],L,delta,slicex1,1), points = [1])
            xyimag = si.quad(func,0,sc.pi,args = (f,x11[x],x22[y],L,delta,slicex1,0), points = [1])
            T[x,y] = 2*xyreal[0] + 2j*xyimag[0]        
    return T

#%% Vector diffraction only for TM mode

def Green0(k, r):
    G = cmath.exp(-1j*k*r)/r/4/sc.pi
    return G

def GN(k, r):
    N = 1j*k + 1/r
    return N

def GM(k, r):
    M = -k**2 + 2j*k/r + 2/r**2
    return M

#%%% cylinder-sphere

def S_D(R0, r, theta, phi, delta):
    D = sqrt(R0**2+ r**2+ delta**2-2*R0*r*sin(theta)*cos(phi)+2*R0*delta*cos(theta))
    return D

def Sphere_cer(phi, k, r, R0, theta, delta, slicex, norm):
    R1 = S_D(R0, r, theta, phi, delta)
    first = -1j*(sc.c*k)*sc.mu_0*sin(theta)*cos(phi)*Green0(k, R1)*slicex
    second= GN(k, R1)*(R0-r*cos(phi)*sin(theta)+delta*cos(theta))/1j/R1*Green0(k, R1)*slicex
    third = -GN(k, R1)*(r*cos(theta)+delta*cos(phi)*cos(theta))/R1*Green0(k, R1)*slicex
    result= [first.real, first.imag, second.real, second.imag, third.real, third.imag]
    return result[norm]

def Sphere_cetheta(phi, k, r, R0, theta, delta, slicex, norm):
    R1 = S_D(R0, r, theta, phi, delta)
    first = -1j*(sc.c*k)*sc.mu_0*cos(theta)*cos(phi)*Green0(k, R1)*slicex
    second= -GN(k, R1)*(r*cos(theta)*cos(phi)+delta*sin(theta))/1j/R1*Green0(k, R1)*slicex
    third = -GN(k, R1)*(R0*cos(phi)-r*sin(theta)+delta*cos(phi)*cos(theta))/R1*Green0(k, R1)*slicex
    result= [first.real, first.imag, second.real, second.imag, third.real, third.imag]
    return result[norm]

def Sphere_cephi(phi, k, r, R0, theta, delta, slicex, norm):
    R1 = S_D(R0, r, theta, phi, delta)
    first = -1j*(sc.c*k)*sc.mu_0*sin(phi)*Green0(k, R1)*slicex
    second= -GN(k, R1)*r*sin(phi)/1j/R1*Green0(k, R1)*slicex
    third = -GN(k, R1)*(R0*sin(phi)*cos(theta)-delta*sin(phi))/R1*Green0(k, R1)*slicex
    result= [first.real, first.imag, second.real, second.imag, third.real, third.imag]
    return result[norm]

def Sphere_chr(phi, k, r, R0, theta, delta, slicex, norm):
    R1 = S_D(R0, r, theta, phi, delta)
    first = -GN(k, R1)*delta*sin(phi)*sin(theta)/R1*Green0(k, R1)*slicex
    second = 0
    third = -1j*sc.c*k*sc.epsilon_0*sin(phi)*sin(theta)*Green0(k, R1)*slicex
    result= [first.real, first.imag, second.real, second.imag, third.real, third.imag]
    return result[norm]

def Sphere_chtheta(phi, k, r, R0, theta, delta, slicex, norm):
    R1 = S_D(R0, r, theta, phi, delta)
    first = -GN(k, R1)*(R0*sin(phi)+delta*sin(phi)*cos(theta))/R1*Green0(k, R1)*slicex
    second = 0
    third = -1j*sc.c*k*sc.epsilon_0*sin(phi)*cos(theta)*Green0(k, R1)*slicex
    result= [first.real, first.imag, second.real, second.imag, third.real, third.imag]
    return result[norm]

def Sphere_chphi(phi, k, r, R0, theta, delta, slicex, norm):
    R1 = S_D(R0, r, theta, phi, delta)
    first = GN(k, R1)*(R0*cos(theta)*cos(phi)+delta*cos(phi))/R1*Green0(k, R1)*slicex
    second = 0
    third = 1j*sc.c*k*sc.epsilon_0*cos(phi)*Green0(k, R1)*slicex
    result= [first.real, first.imag, second.real, second.imag, third.real, third.imag]
    return result[norm]

#%%% cylinder-cylinder

def C_D(Rc, r, z, phi):
    D = sqrt(Rc**2+ r**2+ z**2- 2*Rc*r*cos(phi))
    return D

def Cylinder_cer(phi, k, r, z, Rc, delta, slicex, norm):
    R1 = C_D(Rc, r, z, phi)
    first = -1j*(sc.c*k)*sc.mu_0*cos(phi)*Green0(k, R1)*slicex
    second= GN(k, R1)*(Rc-r*cos(phi))/1j/R1*Green0(k, R1)*slicex
    third = -GN(k, R1)*z*cos(phi)/R1*Green0(k, R1)*slicex
    result= [first.real, first.imag, second.real, second.imag, third.real, third.imag]
    return result[norm]

def Cylinder_cetheta(phi, k, r, z, Rc, delta, slicex, norm):
    R1 = C_D(Rc, r, z, phi)
    first = -1j*(sc.c*k)*sc.mu_0*sin(phi)*Green0(k, R1)*slicex
    second= -GN(k, R1)*r*sin(phi)/(1j*R1)*Green0(k, R1)*slicex
    third = -GN(k, R1)*z*sin(phi)/R1*Green0(k, R1)*slicex
    result= [first.real, first.imag, second.real, second.imag, third.real, third.imag]
    return result[norm]

def Cylinder_cez(phi, k, r, z, Rc, delta, slicex, norm):
    R1 = C_D(Rc, r, z, phi)
    first = 0
    second= GN(k, R1)*z/1j/R1*Green0(k, R1)*slicex
    third = GN(k, R1)*(Rc*cos(phi)-r)/R1*Green0(k, R1)*slicex
    result= [first.real, first.imag, second.real, second.imag, third.real, third.imag]
    return result[norm]

def Cylinder_chr(phi, k, r, z, Rc, delta, slicex, norm):
    R1 = C_D(Rc, r, z, phi)
    first = -GN(k, R1)*z*sin(phi)/R1*Green0(k, R1)*slicex
    second = 0
    third = -1j*sc.c*k*sc.epsilon_0*sin(phi)*Green0(k, R1)*slicex
    result= [first.real, first.imag, second.real, second.imag, third.real, third.imag]
    return result[norm]

def Cylinder_chphi(phi, k, r, z, Rc, delta, slicex, norm):
    R1 = C_D(Rc, r, z, phi)
    first = GN(k, R1)*z*cos(phi)/R1*Green0(k, R1)*slicex
    second = 0
    third = 1j*sc.c*k*sc.epsilon_0*cos(phi)*Green0(k, R1)*slicex
    result= [first.real, first.imag, second.real, second.imag, third.real, third.imag]
    return result[norm]

def Cylinder_chz(phi, k, r, z, Rc, delta, slicex, norm):
    R1 = C_D(Rc, r, z, phi)
    first = GN(k, R1)*Rc*sin(phi)/R1*Green0(k, R1)*slicex
    second = 0
    third = 0
    result= [first.real, first.imag, second.real, second.imag, third.real, third.imag]
    return result[norm]

#%%%sphere-sphere

def ss_D(R0, theta1, theta2, phi, delta):
    D = sqrt(2*R0**2*(1+cos(theta1)*cos(theta2)-sin(theta1)*sin(theta2)*cos(phi))+(delta-R0)**2+2*(delta-R0)*R0*(cos(theta1)+cos(theta2)))
    return D

def Sphere_ser(phi, k, theta1, R0, theta2, delta, slicex, norm):
    R1 = ss_D(R0, theta1, theta2, phi, delta)
    first = -1j*(sc.c*k)*sc.mu_0*(sin(theta2)*cos(theta1)*cos(phi)+sin(theta1)*cos(theta2))*Green0(k, R1)*slicex
    second= -GN(k, R1)*(R0*(sin(theta2)*cos(theta1)*cos(phi)+sin(theta1)*cos(theta2))+(delta-R0)*cos(phi)*sin(theta2))/R1*Green0(k, R1)*slicex
    result= [first.real, first.imag, second.real, second.imag]
    return result[norm]*sc.c

def Sphere_setheta(phi, k, theta1, R0, theta2, delta, slicex, norm):
    R1 = ss_D(R0, theta1, theta2, phi, delta)
    first = -1j*(sc.c*k)*sc.mu_0*(cos(theta2)*cos(theta1)*cos(phi)-sin(theta1)*sin(theta2))*Green0(k, R1)*slicex
    second= -GN(k, R1)*(R0*(cos(phi)+cos(theta2)*cos(theta1)*cos(phi)-sin(theta1)*sin(theta2))+(delta-R0)*cos(phi)*cos(theta2))/R1*Green0(k, R1)*slicex
    result= [first.real, first.imag, second.real, second.imag]
    return result[norm]*sc.c

def Sphere_sephi(phi, k, theta1, R0, theta2, delta, slicex, norm):
    R1 = ss_D(R0, theta1, theta2, phi, delta)
    first = -1j*(sc.c*k)*sc.mu_0*cos(theta1)*sin(phi)*Green0(k, R1)*slicex
    second= -GN(k, R1)*(R0*sin(phi)*(cos(theta1)+cos(theta2))+(delta-R0)*sin(phi))/R1*Green0(k, R1)*slicex
    result= [first.real, first.imag, second.real, second.imag]
    return result[norm]*sc.c

def Sphere_shr(phi, k, theta1, R0, theta2, delta, slicex, norm):
    R1 = ss_D(R0, theta1, theta2, phi, delta)
    second = -1j*(sc.c*k)*sc.epsilon_0*sin(phi)*sin(theta2)*Green0(k, R1)*slicex
    first= -GN(k, R1)*(R0*sin(phi)*sin(theta2)+(delta-R0)*cos(theta1)*sin(theta2)*sin(phi))/R1*Green0(k, R1)*slicex   
    result= [first.real, first.imag, second.real, second.imag]
    return result[norm]*sc.c

def Sphere_shtheta(phi, k, theta1, R0, theta2, delta, slicex, norm):
    R1 = ss_D(R0, theta1, theta2, phi, delta)
    second = -1j*(sc.c*k)*sc.epsilon_0*sin(phi)*cos(theta2)*Green0(k, R1)*slicex
    first = -GN(k, R1)*(R0*sin(phi)*(cos(theta1)+cos(theta2))+(delta-R0)*cos(theta1)*cos(theta2)*sin(phi))/R1*Green0(k, R1)*slicex
    result= [first.real, first.imag, second.real, second.imag]
    return result[norm]*sc.c

def Sphere_shphi(phi, k, theta1, R0, theta2, delta, slicex, norm):
    R1 = ss_D(R0, theta1, theta2, phi, delta)
    second = 1j*(sc.c*k)*sc.epsilon_0*cos(phi)*Green0(k, R1)*slicex
    first= GN(k, R1)*(R0*(cos(theta2)*cos(theta1)*cos(phi)-sin(theta1)*sin(theta2)+cos(phi))+(delta-R0)*cos(phi)*cos(theta1))/R1*Green0(k, R1)*slicex
    result= [first.real, first.imag, second.real, second.imag]
    return result[norm]*sc.c

#%%%sphere-cylinder

def cs_D(R0, theta, rc, z, phi):
    D = sqrt(R0**2+rc**2+(z-R0)**2-2*rc*R0*sin(theta)*cos(phi)+2*R0*(z-R0)*cos(theta))
    return D

def Cylinder_ser(phi, k, theta, R0, rc, z, slicex, norm):
    R1 = cs_D(R0, theta, rc, z, phi)
    first = -1j*(sc.c*k)*sc.mu_0*cos(phi)*cos(theta)*Green0(k, R1)*slicex
    second = -GN(k, R1)*(R0*cos(phi)*cos(theta)+(z-R0)*cos(phi))/R1*Green0(k, R1)*slicex
    result= [first.real, first.imag, second.real, second.imag]
    return result[norm]*sc.c

def Cylinder_sephi(phi, k, theta, R0, rc, z, slicex, norm):
    R1 = cs_D(R0, theta, rc, z, phi)
    first = -1j*(sc.c*k)*sc.mu_0*sin(phi)*cos(theta)*Green0(k, R1)*slicex
    second = -GN(k, R1)*(R0*sin(phi)*cos(theta)+(z-R0)*sin(phi))/R1*Green0(k, R1)*slicex
    result= [first.real, first.imag, second.real, second.imag]
    return result[norm]*sc.c

def Cylinder_sez(phi, k, theta, R0, rc, z, slicex, norm):
    R1 = cs_D(R0, theta, rc, z, phi)
    first = -1j*(sc.c*k)*sc.mu_0*sin(theta)*Green0(k, R1)*slicex
    second = -GN(k, R1)*(R0*sin(theta)-rc*cos(phi))/R1*Green0(k, R1)*slicex
    result= [first.real, first.imag, second.real, second.imag]
    return result[norm]*sc.c

def Cylinder_shr(phi, k, theta, R0, rc, z, slicex, norm):
    R1 = cs_D(R0, theta, rc, z, phi)
    second = -1j*(sc.c*k)*sc.epsilon_0*sin(phi)*Green0(k, R1)*slicex
    first = -GN(k, R1)*(R0*sin(phi)+(z-R0)*sin(phi)*cos(theta))/R1*Green0(k, R1)*slicex
    result= [first.real, first.imag, second.real, second.imag]
    return result[norm]*sc.c

def Cylinder_shphi(phi, k, theta, R0, rc, z, slicex, norm):
    R1 = cs_D(R0, theta, rc, z, phi)
    second = 1j*(sc.c*k)*sc.epsilon_0*cos(phi)*Green0(k, R1)*slicex
    first = GN(k, R1)*(R0*cos(phi)-rc*sin(theta)+(z-R0)*cos(phi)*cos(theta))/R1*Green0(k, R1)*slicex
    result= [first.real, first.imag, second.real, second.imag]
    return result[norm]*sc.c

def Cylinder_shz(phi, k, theta, R0, rc, z, slicex, norm):
    R1 = cs_D(R0, theta, rc, z, phi)
    second = 0
    first = GN(k, R1)*rc*sin(phi)*cos(theta)/R1*Green0(k, R1)*slicex
    result= [first.real, first.imag, second.real, second.imag]
    return result[norm]*sc.c

#%%%matrix calculation

def f_type(typef):
    if typef == 'c2s+er':
        funcs = Sphere_cer 
    elif typef == 'c2s+etheta':
        funcs = Sphere_cetheta 
    elif typef == 'c2s+ephi':
        funcs = Sphere_cephi 
    elif typef == 'c2s+hr':
        funcs = Sphere_chr
    elif typef == 'c2s+htheta':
        funcs = Sphere_chtheta
    elif typef == 'c2s+hphi':
        funcs = Sphere_chphi
    elif typef == 'c2c+er':
        funcs = Cylinder_cer 
    elif typef == 'c2c+etheta':
        funcs = Cylinder_cetheta 
    elif typef == 'c2c+ez':
        funcs = Cylinder_cez
    elif typef == 'c2c+hr':
        funcs = Cylinder_chr
    elif typef == 'c2c+hphi':
        funcs = Cylinder_chphi
    elif typef == 'c2c+hz':
        funcs = Cylinder_chz
    elif typef == 's2s+er':
        funcs = Sphere_ser 
    elif typef == 's2s+etheta':
        funcs = Sphere_setheta 
    elif typef == 's2s+ephi':
        funcs = Sphere_sephi 
    elif typef == 's2s+hr':
        funcs = Sphere_shr
    elif typef == 's2s+htheta':
        funcs = Sphere_shtheta
    elif typef == 's2s+hphi':
        funcs = Sphere_shphi
    elif typef == 's2c+er':
        funcs = Cylinder_ser 
    elif typef == 's2c+ephi':
        funcs = Cylinder_sephi 
    elif typef == 's2c+ez':
        funcs = Cylinder_sez
    elif typef == 's2c+hr':
        funcs = Cylinder_shr
    elif typef == 's2c+hphi':
        funcs = Cylinder_shphi
    elif typef == 's2c+hz':
        funcs = Cylinder_shz
    else:
        print('error in function')
    return funcs

def unit_dif(k, r, R0, theta, delta, slicex, typef):
    funcs = f_type(typef)
    fieldu = []
    if typef[0] == 'c':
        numx = 6
    else:
        numx = 4
    for n in range(numx):
        fieldx = si.quad(funcs, 0, 2*sc.pi, args=(k, r, R0, theta, delta, slicex, n), limit=80, points = [1])
        fieldu.append(fieldx[0])
    if typef[0] == 'c':   
        return [fieldu[0]+1j*fieldu[1], fieldu[2]+1j*fieldu[3], fieldu[4]+1j*fieldu[5]]
    else:
        return [fieldu[0]+1j*fieldu[1], fieldu[2]+1j*fieldu[3]]

def field_matrix(kinds, f, X11, X22, R, delta):
    N = np.size(X11)
    M = np.size(X22)
    Tht = np.zeros((N,M))+1j*np.zeros((N,M))
    Tez = np.zeros((N,M))+1j*np.zeros((N,M))
    Ter = np.zeros((N,M))+1j*np.zeros((N,M))
    k = f*2*sc.pi/sc.c 
    for x in range(N):
        for y in range(M):
            if kinds[0] == 'c':
                slice1 = X11[1]*X11[x]
                Ter[x,y] = unit_dif(k, X11[x], R, X22[y], delta, slice1, kinds)[2]
            elif kinds[0] == 's':
                slice1 = R**2*sin(X11[x])*X11[1]
            else:
                print('error')
            Tht[x,y] = unit_dif(k, X11[x], R, X22[y], delta, slice1, kinds)[0]
            Tez[x,y] = unit_dif(k, X11[x], R, X22[y], delta, slice1, kinds)[1]
    if kinds[0] == 'c':
        return [Tht, Tez, Ter]
    else:
        return [Tht/sc.c, Tez/sc.c]

#%% Tools
    
def cellnumber(f,amplitude):
    lambdatest = sc.c/f
    if amplitude/lambdatest <= 10:
        cnumber = int(100)
    elif amplitude/lambdatest <= 15:
        cnumber = int(round(amplitude/lambdatest)*10)
    elif amplitude/lambdatest <= 22:
        cnumber = int(round(amplitude/lambdatest)*7)
    else:
        cnumber = int(round(amplitude/lambdatest)*5)
    return cnumber

def aperture(N,amplitude,aperturesize,lossfactor):
    aper = np.linspace(1,1,N)
    naper = int(round(aperturesize/amplitude*N))
    aper[0:naper-1] = aper[0:naper-1]*0
    return aper*lossfactor

def hole(N,amplitude,aperturesize,lossfactor):
    '''
    only the field in hole exist
    '''
    aper = np.linspace(1,1,N)
    naper = int(round(aperturesize/amplitude*N))
    aper[naper-1:] = aper[naper-1:]*0
    return aper*lossfactor
    
def power_ratio(X1, X2, field1, field2, L1, L2):
    ## decide the slice, L is decide the whether the mode is sphere
    if L1 != 0:
        slicex1 = L1**2*np.sin(X1)*X1[1]
    else:
        slicex1 = X1*X1[1]
    ##
    if L2 != 0:
        slicex2 = L2**2*np.sin(X2)*X2[1]
    else:
        slicex2 = X2*X2[1]  
    fieldx1 = np.sum(np.abs(field1)**2*slicex1)
    fieldx2 = np.sum(np.abs(field2)**2*slicex2)
    return fieldx2/fieldx1

def eff_p_s(frequency, harmonic, export_r, er, mirror, R_mirr, delta, apert):
    eff_mir = np.linspace(0,0,harmonic)
    for i in range(harmonic):
        light1 = matrix_diff('p2s', frequency[i], export_r[i], mirror, R_mirr, delta) # 第一个出口到镜子的传输矩阵
        firstm0= np.dot(er[i], light1)
        firstm = firstm0*apert
        p_rat1 = power_ratio(export_r, mirror, er, firstm, 0, R_mirr)   #功率传输效率， 0 表示第一个面是平面
        light2 = matrix_diff('s2p', frequency[i], mirror, export_r[i], R_mirr, delta) # 光从镜子反射回来的传输矩阵
        secondm= np.dot(firstm, light2)
        p_rat2 = power_ratio(mirror, export_r, firstm, secondm, R_mirr, 0)
        if p_rat2*p_rat1 >= 1:
            ratio = 0.98
        else:
            ratio = p_rat1*p_rat2
        eff_mir[i] = sqrt(ratio) #功率的传输效率开方之后就是场的幅值的变化量
    return eff_mir

def multi_matrix(matrix1, matrix2, lengthx):
    p2 = np.dot(matrix1[0], matrix2[0])*0
    for i in range(lengthx):
        p1 = np.dot(matrix1[i], matrix2[i])
        p2 = p2+p1
    return p2

def mirror_expansion(matrix, keyx):
    return np.hstack((keyx*matrix[::-1], matrix))

def power_cal(kinds, dist, fielde, fieldh, R):
    fieldhx = fieldh.real-1j*fieldh.imag
    powerd = np.abs((fielde*fieldhx).real)
    if kinds == 's':
        powerx = R**2*np.sin(dist)*dist[1]*powerd*np.pi
        power = np.sum(powerx)
    else:
        powerx = dist*dist[1]*powerd*np.pi
        power = np.sum(powerx)
    return power 
