# -*- coding: utf-8 -*-
"""
Created on Sat Aug  3 18:21:47 2019

@author: biaobin
"""

import scipy.special as ss
import scipy.constants as sc
import scipy.integrate as si
import numpy as np
import math
import models.dispersion as dis
from models.diffraction_matrix import cellnumber
import copy
#import electron as ele
#import matplotlib.pyplot as plt


class distribution:
    
    c=sc.c
    pi=sc.pi
    
    def __init__(self,paramters):
        self.a=paramters[0]
        self.b=paramters[1]
        self.d=paramters[2]
        gama=1+paramters[3]/0.511
        self.beta=dis.gama2beta(gama)
        self.er0=paramters[4]
        self.er1=paramters[5]
        self.f=paramters[6]
        self.N=paramters[7]
        
    def wave_vector(self):
        #这里面的顺序是 k0,kz,kc,kc1,kc2
        k=[]
        k.append(2*self.pi*self.f[0]/self.c)
        k.append(k[0]/self.beta)
        k.append(math.sqrt(abs(k[1]**2-k[0]**2)))
        k.append(math.sqrt(abs(self.er0*(k[0]**2)-k[1]**2)))
        k.append(math.sqrt(abs(self.er1*(k[0]**2)-k[1]**2)))
        return k
    
    def varepsilonr(self,r):
        if r <= self.a:
            er = 1
        elif r <= self.b:
            er = self.er0
        elif r <= self.d:
            er = self.er1
        else:
            er=0
        return er
    
    def k_vec(self,r):
        k = self.wave_vector()
        if r <= self.a:
            k = math.sqrt(abs(k[1]**2-k[0]**2))
        elif r <= self.b:
            k = math.sqrt(abs(self.er0*(k[0]**2)-k[1]**2))
        elif r <= self.d:
            k = math.sqrt(abs(self.er1*(k[0]**2)-k[1]**2))
        else:
            k = 0
        return k
    
    def field_value(self,r):
        kv = self.wave_vector()
        X1 = dis.bessf(0,0,kv[3],self.a,self.b)
        X2 = dis.bessf(0,1,kv[4],self.b,self.d)
        X3 = dis.bessf(0,0,kv[4],self.b,self.d)
        if r<=self.a:
            fieldr = ss.i1(kv[2]*r)/kv[2]*kv[1]
            fieldz = ss.i0(kv[2]*r)
            fieldbt= fieldr*self.c*self.beta*sc.epsilon_0*self.varepsilonr(r)
        elif r<=self.b:
            if self.b < self.d:
                if self.beta>1/math.sqrt(self.er1):
                    A1 = (ss.i0(kv[2]*self.a)*ss.y0(kv[3]*self.b)-ss.y0(kv[3]*self.a)/ss.y0(kv[4]*self.d)*X3*self.f[1])/X1
                    B1 = (-1)*(ss.i0(kv[2]*self.a)*ss.j0(kv[3]*self.b)-ss.j0(kv[3]*self.a)/ss.y0(kv[4]*self.d)*X3*self.f[1])/X1
                else: 
                    A1 = (ss.i0(kv[2]*self.a)*ss.y0(kv[3]*self.b)-ss.y0(kv[3]*self.a)/ss.k0(kv[4]*self.d)*X2*self.f[1])/X1
                    B1 = (-1)*(ss.i0(kv[2]*self.a)*ss.j0(kv[3]*self.b)-ss.j0(kv[3]*self.a)/ss.k0(kv[4]*self.d)*X2*self.f[1])/X1
                fieldr = (A1*ss.j1(kv[3]*r)+B1*ss.y1(kv[3]*r))/kv[3]*kv[1]
                fieldz = A1*ss.j0(kv[3]*r)+B1*ss.y0(kv[3]*r)
                fieldbt= fieldr*self.c*self.beta*sc.epsilon_0*self.varepsilonr(r)
            else:
                fieldr = -self.f[1]*dis.bessf(1,0,kv[3],r,self.b)/kv[3]*kv[1]
                fieldz = self.f[1]*dis.bessf(0,0,kv[3],r,self.b)
                fieldbt= fieldr*self.c*self.beta*sc.epsilon_0*self.varepsilonr(r)
        elif r<=self.d:
            if self.beta>1/math.sqrt(self.er1):
                fieldr = -self.f[1]*dis.bessf(1,0,kv[4],r,self.d)/ss.y0(kv[4]*self.d)/kv[4]*kv[1]
                fieldz = self.f[1]*dis.bessf(0,0,kv[4],r,self.d)/ss.y0(kv[4]*self.d)
                fieldbt= fieldr*self.c*self.beta*sc.epsilon_0*self.varepsilonr(r)
            else:
                fieldr = self.f[1]*dis.bessf(1,1,kv[4],r,self.d)/ss.k0(kv[4]*self.d)/kv[4]*kv[1]
                fieldz = self.f[1]*dis.bessf(0,1,kv[4],r,self.d)/ss.k0(kv[4]*self.d)
                fieldbt= fieldr*self.c*self.beta*sc.epsilon_0*self.varepsilonr(r)
        else:
            fieldr = 0
            fieldz = 0
            fieldbt= 0
        return [fieldr,fieldz,fieldbt]
    #%% Each Factors
    def energydensity(self,r):
        field = self.field_value(r)
        err = self.varepsilonr(r)*sc.epsilon_0
        k0 = self.wave_vector()  ###注意这边要与前面的场的kz/kc相互符合
        Ue = err*field[0]**2 + err*field[1]**2 + self.varepsilonr(r)*err*(k0[0]**2/k0[1]**2)*field[0]**2
        return Ue*r*self.pi*1e18/2  ###这边数值太小了，导致误差很大，所以提升9个数量级，这个要注意
    
    
    def powerdensity(self,r):
        field = self.field_value(r)
        err = self.varepsilonr(r)*sc.epsilon_0
        pd = self.beta*err*self.c*field[0]**2
        return pd*r*self.pi*1e18  ###这边数值太小，必须乘上1e9, 不然积分时候的误差过大
    
    def power_export(self, field_mag):
        return si.quad(self.powerdensity, 0, self.d)[0]/1e18*field_mag**2#单位为W
    
    def dielossdensity(self,r):
        field = self.field_value(r)
        err = self.varepsilonr(r)*sc.epsilon_0
        tand = 2e-4 ##熔融石英
        dieloss = sc.pi*(2*sc.pi*self.f[0])*err*tand*(field[0]**2+field[1]**2)
        return dieloss*r/2   #注意这里积分的时候会出现1/2
        
    
    def lossfactor(self):
        power = si.quad(self.powerdensity,0,self.d)
        energy = si.quad(self.energydensity,0,self.d)
        vg = power[0]/energy[0]/sc.c
        lossf=1/(energy[0]-power[0]/self.beta/sc.c)/4*1e9
        return [lossf,vg]
    
    def metalloss(self):
        field = self.field_value(self.d)
        kz = self.wave_vector()
        sig = 5.8137e7 ##电导率取的是铜的
        err = self.er1*sc.epsilon_0
        Rs = math.sqrt(sc.pi*self.f[0]*sc.mu_0/sig)
        Htheta = err*2*sc.pi*self.f[0]/kz[1]*field[0]
        mloss = sc.pi*self.d*Rs*Htheta**2
        return mloss/2
    
    def alpha0(self):
        dloss = si.quad(self.dielossdensity,self.a,self.d)
        Ploss = self.metalloss()+ dloss[0]
        Power = si.quad(self.powerdensity,0,self.d)
        alpha = Ploss/2/Power[0]
        return alpha
    
    def alphac(self):
        alpha00 = self.alpha0()
        lossf = self.lossfactor()
        vg = lossf[1]
        factor = self.beta/(self.beta-vg)-1
        return alpha00*factor
    
    def impedience(self):
        dloss1 = si.quad(self.dielossdensity, self.a, self.d)
        Zs     = 1/(self.metalloss() + dloss1[0])
        return Zs
    
    def quality(self):
        '''
        the quality factor Zs/Q
        '''
        energy = si.quad(self.energydensity, self.a, self.d)
        qu_f   = 1/energy[0]/(2*sc.pi*sc.c)
        return qu_f
        

    #%% Transverse distribution
    
    def r_dist(self):
        return np.linspace(0,self.d,self.N,endpoint=False)

    
    def er_dist(self):
        radi = np.linspace(0,self.d,self.N,endpoint=False)
        er_dist1 = np.linspace(0,0,self.N)
        for i in range(self.N):
            field = self.field_value(radi[i])
            er_dist1[i] = field[0]
        return er_dist1
    
    def ez_dist(self):
        radi = self.r_dist()
        ez_dist1 = np.linspace(0,0,self.N)
        for i in range(self.N):
            field = self.field_value(radi[i])
            ez_dist1[i] = field[1]
        return ez_dist1
    
    def bt_dist(self):
        radi = self.r_dist()
        bt_dist1 = np.linspace(0,0,self.N)
        for i in range(self.N):
            field = self.field_value(radi[i])
            bt_dist1[i] = field[2]
        return bt_dist1
 
    def fieldr_dist(self, mag):
        m = [self.bt_dist()*mag, self.ez_dist()*mag, self.er_dist()*mag]
        return m

#%% tools 
    
def one2two(dist,rkey=0):
       ###这个采用了矩阵找交点的方法，但是只是适合单点的情况，另外找点这里面莫名其妙会出现两个值。。
       ###但是必须采用这里面定义的范式即r,z,field,field,0为径向场，1，为纵向场
    lens=len(dist[0])
    unit=np.ones((1,lens))
    radius0=np.array(dist[0])
    radius0=radius0.reshape((1,lens))
    
    dist3=np.zeros((int(2*lens),int(2*lens)))
    
    x=np.hstack((-1*dist[0][::-1],dist[0]))
    y=np.hstack((-1*dist[0][::-1],dist[0]))
    rmax=np.max(x)
    
    for n in range(int(2*lens)):
        for m in range(int(2*lens)):
            radius=math.sqrt(x[n]**2+y[m]**2)
            if radius >= rmax:
                dist3[n,m]=0
            else:
                error1=np.abs(np.subtract(radius0,radius*unit))
                error2=np.min(error1)
                index1=np.argwhere(error1==error2)
                ind=int(index1[0,1])
                dist3[n,m]=dist[rkey+2][ind]
    return dist3
    
def multimode_field(paramters0):
    dispers = dis.dispersion(paramters0)
#    frequency = np.linspace(0,0,harmonic)
    harmonic = dispers.mode_max()
    paramters=copy.deepcopy(paramters0)
    r = []
    m = []
    for i in range(harmonic):
        fmag = dispers.wakeinfo(i+1)  
        Nguide = cellnumber(fmag[0],paramters[2])
        paramters[6] = fmag
        paramters[7] = Nguide
        distf = distribution(paramters)
        powerx = distf.power_export(1)
        rd = distf.r_dist()
        er0 = distf.fieldr_dist(1/np.sqrt(powerx))
        ####
#        frequency[i] = fmag[0]
        r = rd
        m.append(er0)
    return r, m #frequency
   
def multimode_guide(paramters):
    dispers = dis.dispersion(paramters)
    param2 = copy.deepcopy(paramters)
    harmonic = dispers.mode_max()
    frequency = np.linspace(0,0,harmonic)
    kappa = np.linspace(0,0,harmonic)
    vg = np.linspace(0,0,harmonic)
    loss0 = np.linspace(0,0,harmonic)
    lossc = np.linspace(0,0,harmonic)
    for i in range(harmonic):
        fmag = dispers.wakeinfo(i+1) 
        Nguide = cellnumber(fmag[0],paramters[2])
        param2[6] = fmag
        param2[7] = Nguide
        distf = distribution(param2)
        factor = distf.lossfactor()
        l0  = distf.alpha0()
        lc  = distf.alphac()
        ####
        frequency[i] = fmag[0]
        kappa[i] = factor[0] ##单位是V/nC/m
        vg[i] = factor[1]  ##单位是1
        loss0[i] = l0
        lossc[i] = lc
    return frequency, kappa, vg, loss0, lossc

def multimode_loss(alpha, L):
    eff_g = np.linspace(0,0,len(alpha))
    for i in range(len(alpha)):
        eff_g[i] = math.exp(-1*alpha[i]*L)
    return eff_g
    
if __name__=='__main__':
    paramter=[0.4e-3,0.55e-3,0.55e-3,10,3.8,1.02,2e12,2000]
    disper=dis.dispersion(paramter)
    f=disper.wakeinfo(1)
    paramter2=paramter
    paramter2[6]=f
    paramter2[7]=100
    dist1=distribution(paramter2)
    lossfactor=dist1.lossfactor()
    dist=dist1.radi_dist(1.0)
#    dist3=one2two(dist)
    #plt.imshow(dist3,extent=[-1.5,1.5,-1.5,1.5],origin='lower')
###200以上的，画图的边缘才不会太过明显，这里的用contour好一些
    #plt.contourf(dist3,30)
#    plt.plot(dist[0,:],dist[1,:])
