# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 16:03:35 2019

@author: Shimin
"""

from scipy.constants import c, pi
from scipy.special import erfinv
# import numpy as np
from random import shuffle
from ghalton import Halton
from math import sqrt, log, exp
from matplotlib.pyplot import hist#, plot
#from numba import jit

def halton_list(num):
    rand = Halton(5)               # halton 序列生成器
    rand0= rand.get(num)
    shuffle(rand0)  # 打乱顺序
    rand2= [rand0[i][2] for i in range(num)]  # 挑选第2串序列
    return rand2

#%% Distribution
def uniform_dist(L, num:int):
    x = [((i+0.5)/num-0.5)*L for i in range(num)] 
    return x

def gauss_dist(sig, num:int):
    g = lambda x:sig*sqrt(2)*erfinv(2*x-1) #erfinv 误差函数的反函数，用于纵向的分布
    y = [g((i+0.5)/num) for i in range(num)]
    return y

def rad_rand_dist(R, num:int):
    rand2 = halton_list(num)
    rad  = [sqrt(x)*R for x in rand2]# 生成径向分布的list
    return rad

def gauss_rand_dist(sig, num:int):
    rand = halton_list(num)
    g = lambda x:sig*sqrt(-2*log(x))      # 使用极坐标变化的方法产生径向的高斯分布更加简单
    gr = [g(i) for i in rand]
    return gr

#%% coherent factor
def coherent_factorT(f,sigr,beta):
    k = 2*pi*f/c  # 横向的群聚因子
    gama = 1/sqrt(1-beta**2)
    return exp(-(k*sigr/beta/gama)**2/2)

def coherent_factorZ(f,sigmaz):
    k = 2*pi*f/c  #纵向的群聚因子，单位是长度
    return exp(-k**2*sigmaz**2/2)

#%% beta-gamma
def gama2beta(gamma):
    return sqrt(gamma**2-1)/gamma

def beta2gama(beta):
    return 1/sqrt(1-beta**2)

def energy2gama(energy0):
    return 1+energy0/0.511

def gama2energy(gamma):
    return (gamma-1)*0.511

if __name__ == '__main__':
#    from numpy.random import uniform, normal
    Nele = 10000
    # length = 0.08
    # ele = rad_rand_dist(1,Nele)
    # ele = uniform(0, 1, Nele)
    # ele1 = gauss_dist(1, Nele)
    # ele2= halton_list(Nele)
    # ele3 = normal(0, 1, Nele)
    ele4 = gauss_rand_dist(1, Nele)
    x,y,z = hist(ele4,bins=250)

    
