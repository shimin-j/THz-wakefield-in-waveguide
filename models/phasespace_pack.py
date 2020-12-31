# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 16:55:48 2019

@author: Shimin
"""
import numpy as np
from scipy.constants import c, pi
from scipy.special import i0, i1
from cmath import exp
from math import sqrt
# import copy
import models.electron as ele
#import dispersion as disp
# from numba import jit, vectorize
#
#%% tools
def slice_gen(fre_max, beta, vg_list, N):
    dz = c*beta/fre_max/N                         # 根据最大的频率划分前进的步长
    dt = dz/c           
    dxi= dt*(beta*c-vg_list*c)# 根据前进的步长来确定不同频率下的场点划分
    return dt, dz, dxi                       

def k_z(fre_list, alpha_list, beta):
    kz0= 2*pi*fre_list/c/beta+1j*alpha_list
    return kz0

def k_t(fre_list, beta):
    kt0 = 2*pi*fre_list/c*sqrt(1/beta**2-1)
    return kt0

def fm2en(fre_max, ele_lent):
    T = 1/ fre_max
    if ele_lent<= 10*T:
        eN = 200*int(ele_lent//T)
    elif ele_lent<= 50*T:
        eN = 80 * int(ele_lent//T)
    else:
        eN = 20* int(ele_lent//T)
    if eN > 10000:
        print('Electron number is too large!')
    return eN

def field_add(field_group, slicez, slicexi):
    field_dist = np. arange(0, field_group[0][0][-1], slicez)
    field_add  = 0*field_dist
    num = len(field_dist)
    harmonic = len(field_group)
    for i in range(num):
        for n in range(harmonic):
            x = int((field_group[n][0][-1]-field_dist[i])//slicexi[n]+2)
            if x < len(field_group[n][0]):
                field_add[i] +=  field_group[n][1][-x]
            else:
                field_add[i] += 0
    return [field_dist, field_add]

def field_g(elengthz, slicexi):
    field_group = []
    for xi in slicexi:
        # 全范围的计算对于中间断点的图像显示是不友好的
        field1    = np.arange(0, elengthz+xi, xi)
        num       = len(field1)
        field_pos = field1-np.max(field1) # 生成与电子束对应的场
        field_amp = np.linspace(0, 0, num) # 这一类会转为复数，所以最好分离开
        # 最快为 0.18s,直接用numpy生成
        # 通过循环加numba加速，时间为3s
        # 通过列表表达式得到的时间为5s           
        field_group.append([field_pos, field_amp])
    return num, field_group


def ele_g(length_L, length_R, Q, gama, num, type_L, type_R):
    q = Q/num    # Q为nC
    energy = [gama for i in range(num)]
    dr     = [0 for i in range(num)]
    if type_L == 'uniform':
        z_dist = ele.uniform_dist(length_L, num)
    elif type_L == 'gauss':
        z_dist = ele.gauss_dist(length_L,num)
    else:
        pass
    
    if type_R == 'uniform':
        r_dist = ele.rad_rand_dist(length_R, num)
    elif type_R == 'gauss':
        r_dist = ele.gauss_rand_dist(length_R, num)
    else:
        pass
    ele_group = np.array([z_dist, energy, r_dist, dr])
    ele_group[0] = ele_group[0]-np.max(ele_group[0])
    return q, ele_group


def probe_g(pz, field_group, slicexi):
    harmonic = len(field_group)
    px       = 0
    if pz >= field_group[0][0][-1]:
        pass
    else:
        for i in range(harmonic):
            x = int((field_group[i][0][-1]-pz)//slicexi[i]+2)
            if x < len(field_group[i][0]):
                px += field_group[i][1][-x]
            else:
                px += 0
    return px

#%% process 

def e_step(ele_phase, field_group, params, beta, slicez, slicet, slicexi):
    fre      = params[0]
    kappa    = params[1]
    vg       = params[2]
    alpha0   = params[3]
    kz       = k_z(fre, alpha0, beta)
    harmonic = len(fre)
    power    = np.linspace(0,0,harmonic)
    # print(kz)
    ele_phase[0] = ele_phase[0]+slicez
    for i in range(harmonic):
        field_group[i][0] += slicet* c* vg[i]
        new_slice         = field_group[i][0][-1]+ slicexi[i]
        field_group[i][0] = np.append(field_group[i][0], new_slice)
        field_group[i][1] = np.append(field_group[i][1], 0)
        field_group[i][1] = field_group[i][1]* exp(1j*kz[i]*slicexi[i])
        
        E_max = np.max(np.abs(field_group[i][1]))
        power[i] = E_max**2*vg[i]*c/(4*kappa[i]*(1-vg[i]/beta))/1e15
    # print(kz)
    return ele_phase, field_group, power



def field_step_1D(ele_phase, field_group, params, beta, q, slicez, slicet, slicexi):
    fre      = params[0]
    kappa    = params[1]
    alpha0   = params[3]# 主要是为了和waveguide里面匹配
    kz       = k_z(fre, alpha0, beta)
    harmonic = len(fre)
    Nele     = len(ele_phase[0])
    
    for m in range(Nele):
        dgama = 0
        for n in range(harmonic): # 计算每个模式上电子对与场的贡献
            zn = int((ele_phase[0][-1]-ele_phase[0][m])//slicexi[n]+1+1)
            zs = (ele_phase[0][-1]-ele_phase[0][m])%slicexi[n]
            if ele_phase[0][m] <= 0:
                pass
            elif zs == 0:
                field_group[n][1][-zn]+= -kappa[n]*q
            else:
                dzx = ele_phase[0][m]-field_group[n][0][-zn]
                field_group[n][1][-zn]+= -2*kappa[n]*q*exp(1j*kz[n]*dzx)#slicez-zs
            dgama = dgama + 1/0.511/1e6*slicez*field_group[n][1][-zn].real
            
        ele_phase[1][m]+= dgama
        ds = c*sqrt(1-1/ele_phase[1][m]**2)*slicet  #将gamma转化为beta重新计算走的距离
        ele_phase[0][m]+= ds-slicez
    # print(kz)
    return ele_phase, field_group

def field_step_2D(ele_phase, field_group, params, beta, q, slicez, slicet, slicexi):
    fre      = params[0]   # Hz
    kappa    = params[1]    #V/m/nC
    alpha0   = params[3]
    harmonic = len(fre)
    Nele     = len(ele_phase[0])
    kz       = k_z(fre, alpha0, beta)
    kt       = k_t(fre, beta)
    harmonic = len(fre)
    Nele     = len(ele_phase[0])
    
    for m in range(Nele):
        dgama = 0
        drf   = 0
        betam = sqrt(ele_phase[1][m]**2-1)/ele_phase[1][m]  # 将每个电子的beta重新标定一下
        for n in range(harmonic):
            zf = i0(kt[n]*ele_phase[2, m])
            rf = i1(kt[n]*ele_phase[2, m])*sqrt(1-betam**2) 
            zn = int((ele_phase[0][-1]-ele_phase[0][m])//slicexi[n]+1+1) # 场的位置需要落后于电子位置
            zs = (ele_phase[0][-1]-ele_phase[0][m])%slicexi[n]
            if ele_phase[0][m] <= 0:
                pass
            elif zs == 0:
                field_group[n][1][-zn]+= -kappa[n]*q*zf      # 只计算到轴上的场
            else:
                dzx = ele_phase[0][m]-field_group[n][0][-zn]
                field_group[n][1][-zn]+= -2*kappa[n]*q*zf*exp(1j*kz[n]*dzx)
            dfact =  1/0.511/1e6*slicez*field_group[n][1][-zn]  
            dgama += dfact.real*zf           # 对于电子纵向的影响
            drf   += dfact.imag*rf/betam     # 对电子径向的影响
        ds = c*sqrt(1-1/ele_phase[1][m]**2)*slicet
        dr = ele_phase[3][m]/betam/ele_phase[1][m]*slicez
        ele_phase[1, m]+= dgama
        ele_phase[3, m]+= drf
        ele_phase[0, m]+= ds-slicez
        ele_phase[2, m] = abs(ele_phase[2][m]+dr)
        
    return ele_phase, field_group

def phasespace(params_guide, params_ele, params_slice, typelist):
    fre     = params_guide[0]
    vg      = params_guide[2]
    distance= params_guide[4]
    harmonic= len(fre)
    
    Q       = params_ele[0]    
    energy  = params_ele[1]
    ele_lent= params_ele[2]
    ele_lenr= params_ele[3]
    gama    = ele.energy2gama(energy)
    beta    = ele.gama2beta(gama)
    ele_lenz= ele_lent*c*beta
    
    Nxi     = params_slice[0]
    Nele    = params_slice[1]
    snap    = params_slice[2]
    probe_z = params_slice[3]
    
    type_L  = typelist[0]
    type_R  = typelist[1]
    typeD   = typelist[2]
    
    dt, dz, dxi          = slice_gen(fre[-1], beta, vg, Nxi)
    micro_q, ele_group   = ele_g(ele_lenz, ele_lenr, Q, gama, Nele, type_L, type_R)
    run_num              = int(distance//dz)+1
    field_num,field_group= field_g(ele_lenz, dxi)
    
    snap_shot            = run_num//snap
    
    power = np.zeros((harmonic, run_num))
    probe = np.zeros((2, run_num))
    field = []
    electron = []
    for i in range(run_num):
        ele_group, field_group, powerx = e_step(ele_group, field_group, params_guide, beta, dz, dt, dxi)
        power[:, i] = powerx
        if typeD == '1d':
            ele_group, field_group = field_step_1D(ele_group, field_group, params_guide, beta, micro_q, dz, dt, dxi)
        else:
            ele_group, field_group = field_step_2D(ele_group, field_group, params_guide, beta, micro_q, dz, dt, dxi)
        probe[0][i] = dt*i
        probe[1][i] = probe_g(probe_z, field_group, dxi)
        # 这里可以按照等间距的时间间隔，将场和电子文件保存下来
        if i % snap_shot == 0:
            field.append(field_group)
            electron.append(ele_group)
            print(i//snap_shot)
        else:
            pass
    if i% snap != 0: #将最后一个文件保存下来
        field.append(field_group)
        electron.append(field_group)
    else:
        pass
    
    return field, electron, power, probe
        
        
if __name__ == '__main__':
    from distribution import multimode_guide
    from time import time
#    from matplotlib.pyplot import plot
    time0 = time()
    paramters = [0.5e-3, 0.55e-3, 0.55e-3, 5.4, 3.8, 1.02, 1e12, 1000]#[ra, rb, rex, energy, er1, er2, f_max, N]
    fre, kappa, vg, a0, ac = multimode_guide(paramters)
    params_guide = [fre, kappa, vg, ac, 0.25]  #[fre, kappa, vg, ac, L]
    params_ele   = [1, 5.4, 10e-12, 0.25e-3]   #[Q, energy, len_t, len_r]
    params_slice = [50, 2000, 10, 0.1]         #[Nxi, Nele, snap, probez]
    params_type  = ['uniform', 'uniform','1d'] #[type_L, type_R, type_1/2]
    
    fieldx, electronx, power, probe = phasespace(params_guide, params_ele, params_slice, params_type)
    # fre     = params_guide[0]
    # vg      = params_guide[2]
    # distance= params_guide[4]
    # harmonic= len(fre)
    
    # Q       = params_ele[0]    
    # energy  = params_ele[1]
    # ele_lent= params_ele[2]
    # ele_lenr= params_ele[3]
    # gama    = ele.energy2gama(energy)
    # beta    = ele.gama2beta(gama)
    # ele_lenz= ele_lent*c*beta
    
    # Nxi     = params_slice[0]
    # Nele    = params_slice[1]
    
    # type_L  = params_type[0]
    # type_R  = params_type[1]
    # typeD   = params_type[2]
    # # 用时0.7s
    
    # time1 = time()
    # dt, dz, dxi        = slice_gen(fre[-1], beta, vg, Nxi)
    # time11 = time()
    # field_num,field_group = field_g(ele_lenz, dxi)
    # time12 = time()
    # # Nele = 2*field_num
    # micro_q, ele_group    = ele_g(ele_lenz, ele_lenr, Q, gama, Nele, type_L, type_R)
    
    
    
    # time2 = time()
    
    # run_num = int(distance//dz)+1
    # power = np.zeros((harmonic, run_num))
    # for i in range(run_num):
    #     # time3 = time()
    #     ele_group, field_group, powerx= e_step(ele_group, field_group, params_guide, beta, dz, dt, dxi)
    #     power[:, i] = powerx
    #     # time3= time()
    #     ele_group, field_group = field_step_2D(ele_group, field_group, params_guide, beta, micro_q, dz, dt, dxi)
    #     if i%(run_num//10) ==0:
    #         print(i/(run_num//10))
    #     else:
    #         pass
    # time4 = time()
    # # print(time4-time3)
    
    # # print(time11-time1)
    # # print(time12-time11)
    # # print(time2-time12)
    # # print(time3-time2)
    # # print(time4-time3)
    # print(time4-time2)
    
    # # ele_group, field_group, power = phasespace(params_guide, params_ele, params_slice, params_type)
    # plot(field_group[0][0], np.real(field_group[0][1]))
