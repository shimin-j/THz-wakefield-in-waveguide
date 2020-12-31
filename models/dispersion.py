# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 15:28:23 2019

@author: biaobin
"""

import numpy as np
import scipy.special as ss
import scipy.constants as sc
import math


def bessf(key1,key2,k,r1,r2):
        # key2 表示是否有虚数，key1表示是否是有导数
    if key2==1 and key1==1:
        bess=ss.i1(k*r1)*ss.k0(k*r2)+ss.k1(k*r1)*ss.i0(k*r2)
    elif key2==1 and key1==0:
        bess=ss.i0(k*r1)*ss.k0(k*r2)-ss.k0(k*r1)*ss.i0(k*r2)
    elif key2==0 and key1==1:
        bess=-ss.j1(k*r1)*ss.y0(k*r2)+ss.y1(k*r1)*ss.j0(k*r2)
    elif key2==0 and key1==0:
        bess=ss.j0(k*r1)*ss.y0(k*r2)-ss.y0(k*r1)*ss.j0(k*r2)
    else:
        print('error')
    return bess

def gama2beta(gamma):
    return np.sqrt(gamma**2-1)/gamma

def beta2gama(beta):
    return 1/np.sqrt(1-beta**2)

def energy2gama(energy0):
    return 1+energy0/0.511

def gama2energy(gamma):
    return (gamma-1)*0.511

def energy2beta(energy):
    return gama2beta(energy2gama(energy))

def beta2energy(beta):
    return gama2energy(beta2gama(beta))

def findpoint(fun,N,sllice):
    f=np.zeros((2,N))
    f2=np.zeros((2,101))
    ff=[]
    for i in range(N):
        f[0][i]=(i+1)*sllice #最小刻度是GHz
        med=fun(f[0][i])####
        f[1][i]=med[0]
        if i>1 and (f[1][i]*f[1][i-1]<0) and ((f[1][i-2]-f[1][i-1])*(f[1][i-1]-f[1][i])>0):
            f2[0][0]=f[0][i]   
            f2[1][0]=f[1][i]
            for m in range(100):
                f1=f[0][i-1]+(f[0][i]-f[0][i-1])/100*(m+1)
                med2=fun(f1)####
                f2[0][m+1]=f1
                f2[1][m+1]=med2[0]
                if m>0 and f2[1][m+1]*f2[1][m]<0 and ((f2[1][m+1]-f2[1][m])*(f2[1][m]-f2[1][m-1])>0): 
                    # 这里居然出现了阶越点了。
                    ff.append(f1)
                else:
                    pass
        else:
            pass
    return ff

class dispersion:
    c=sc.c
    pi=sc.pi
    
    def __init__(self,paramters):
        self.a=paramters[0]
        self.b=paramters[1]
        self.d=paramters[2]
        self.energy=paramters[3]
        self.er0=paramters[4]
        self.er1=paramters[5]
        self.fmax=paramters[6]
        self.N=paramters[7]
        gama=1+abs(self.energy)/0.511
        if paramters[3] >= 0:
            self.beta=math.sqrt(gama**2-1)/gama
        else:
            self.beta= abs(paramters[3])
        
    def discribe(self):
        return '这里计算的是一个r1={}m,r2={}m,r3={}m的介质波导,\
    其中，第一层介质的介质常数为{}，第二层介质常数为{}，电子的能量为{}MeV,\
    当输入精度为GHz时，输出误差在0.1GHz以内'.format(self.a,self.b,self.d,self.er0,self.er1,self.energy)
    

    
    def disper_eq(self,fre):
        
        
        k=2*self.pi*fre/self.c
        kz=k/self.beta
        kc=math.sqrt(abs(kz**2-k**2))
        kc1=math.sqrt(abs(self.er0*(k**2)-kz**2))
        kc2=math.sqrt(abs(self.er1*(k**2)-kz**2))
        
        X1=bessf(0,0,kc1,self.a,self.b)
        X2=bessf(1,0,kc1,self.a,self.b)
        X3=bessf(1,0,kc1,self.a,self.a)
        X4=bessf(1,0,kc1,self.b,self.b)
        X5=bessf(1,0,kc1,self.b,self.a)
        
        if self.beta>= 1:
            
            X6=bessf(1,0,kc2,self.b,self.d)
            X7=bessf(0,0,kc2,self.b,self.d)
            
            M11=self.er0*X2/X1*ss.j0(kc*self.a)/kc1+1/kc*ss.j1(kc*self.a)
            M12=-self.er0*X3/X1*X7/ss.y0(kc2*self.d)/kc1
            M21=self.er0*X4/X1*ss.j0(kc*self.a)/kc1
            M22=-self.er0*X5/X1*X7/ss.y0(kc2*self.d)/kc1-self.er1/kc2*X6/ss.y0(kc2*self.d)
        elif self.beta>=1/math.sqrt(self.er1):
            
            X6=bessf(1,0,kc2,self.b,self.d)
            X7=bessf(0,0,kc2,self.b,self.d)
            
            M11=self.er0*X2/X1*ss.i0(kc*self.a)/kc1+1/kc*ss.i1(kc*self.a)
            M12=-self.er0*X3/X1*X7/ss.y0(kc2*self.d)/kc1
            M21=self.er0*X4/X1*ss.i0(kc*self.a)/kc1
            M22=-self.er0*X5/X1*X7/ss.y0(kc2*self.d)/kc1-self.er1/kc2*X6/ss.y0(kc2*self.d)
        else:
            
            X6=bessf(1,1,kc2,self.b,self.d)
            X7=bessf(0,1,kc2,self.b,self.d)
            
            M11=self.er0*X2/X1*ss.i0(kc*self.a)/kc1+1/kc*ss.i1(kc*self.a)
            M12=-self.er0*X3/X1*X7/ss.k0(kc2*self.d)/kc1
            M21=self.er0*X4/X1*ss.i0(kc*self.a)/kc1
            M22=-self.er0*X5/X1*X7/ss.k0(kc2*self.d)/kc1+self.er1/kc2*X6/ss.k0(kc2*self.d)
            
        f=M11*M22-M12*M21
        if M12 != 0:
            mag=-1*M11/M12
        else:
            mag=ss.i0(kc*self.a)/bessf(0,0,kc1,self.a,self.b)
        
        return [f,mag]
        
    
    def wakeinfo(self,harmonic):
        f3=findpoint(self.disper_eq,self.N,self.fmax/self.N)
        f4=f3[harmonic-1]
        mag1=self.disper_eq(f4)
        return [f4,mag1[1]]
        
    def mode_max(self):
        fx = findpoint(self.disper_eq,self.N,self.fmax/self.N)
        return len(fx)
    
    def groupvelocity(self,harmonic):
        f1=self.wakeinfo(harmonic)
        beta1=self.beta
        self.beta=0.99*self.beta
        f2=self.wakeinfo(harmonic)
        betag=(f2[0]-f1[0])/(f2[0]/beta1/0.99-f1[0]/beta1)
        return betag
    
    def pulselen(self, harmonic, L):
        vg = self.groupvelocity(harmonic)
        pulselength = L/self.c/vg-L/self.c/self.beta
        return pulselength
        
        
####波导管中的各个参数的计算都需要算进去，算是对于我自己的一个经验的整合
     
                                
if __name__=='__main__':
    paramter=[0.25e-3,0.35e-3,0.35e-3,10,3.8,1.02,2e12,2000]
    disper=dispersion(paramter)
    f=disper.wakeinfo(1)
    betag=disper.groupvelocity(1)
    print(f)
    

    
                    
  
        
    
    
