# -*- coding: utf-8 -*-

"""
Module implementing MainWindow.
"""

from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QMainWindow,  QApplication

from Ui_GUI_test4 import Ui_MainWindow
import models.distribution as distf
from models.phasespace_pack import *
import models.electron as ele
import models.diffraction_matrix as difm
import numpy as np
from copy import deepcopy
from scipy.constants import c, pi
from pandas import DataFrame
from qtpandas.models.DataFrameModel import DataFrameModel

class MainWindow(QMainWindow, Ui_MainWindow):
    """
    Class documentation goes here.
    """
    def __init__(self, parent=None):
        """
        Constructor
        
        @param parent reference to the parent widget
        @type QWidget
        """
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)
        widget1 = self.pandastabelwidget
        self.model = DataFrameModel()
        widget1.setViewModel(self.model)
        #下拉选项只有选项改变的时候才会触发，所以必须提前赋值
        self.distzx = 'uniform'  
        self.distrx = 'uniform'
        self.harmonic1x = 0
        self.harmonic2x = 0
        self.harmonic3x = 0
        self.field_type1d = 'probe'
        self.field_type2d = 'probe'
        self.electron_type2d = 'Energy'
        self.Tfield = 'Ez'
        self.farfieldx = 'E_theta'
    
    def str2floatx(self,  strx):
        if strx =='':
            f = 0
        else:
            f = float(strx)
        return f
    #============================guide============================
    @pyqtSlot(str)
    def on_ra_textChanged(self, p0):
        self.rax = self.str2floatx(p0)*1e-3
    
    @pyqtSlot(str)
    def on_rb_textChanged(self, p0):
        self.rbx = self.str2floatx(p0)*1e-3
    
    @pyqtSlot(str)
    def on_re1_textChanged(self, p0):
        self.rex = self.str2floatx(p0)*1e-3
    
    @pyqtSlot(str)
    def on_L_textChanged(self, p0):
        self.Lx = self.str2floatx(p0)*1e-3
    
    @pyqtSlot(str)
    def on_eps1_textChanged(self, p0):
        self.eps1x = self.str2floatx(p0)
    
    @pyqtSlot(str)
    def on_eps2_textChanged(self, p0):
        self.eps2x = self.str2floatx(p0)
    
    #================================electron============================
    @pyqtSlot(str)
    def on_Q_textChanged(self, p0):
        self.Qx = self.str2floatx(p0)      # 单位nC
    
    @pyqtSlot(str)
    def on_Energy_textChanged(self, p0):
        self.energyx = self.str2floatx(p0)    #单位MeV
        self.gama = ele.energy2gama(self.energyx) 
        self.beta   = ele.gama2beta(self.gama)
    
    @pyqtSlot(str)
    def on_sigt_textChanged(self, p0):
        self.sigtx =self.str2floatx(p0)*1e-12   #时间
        self.sigzx = self.sigtx*self.beta*c        #位置
    
    @pyqtSlot(str)
    def on_sigr_textChanged(self, p0):
        self.sigrx = self.str2floatx(p0)*1e-3
    
    @pyqtSlot(str)
    def on_distz_currentTextChanged(self, p0):
        self.distzx = p0
        print(self.distzx)
    
    @pyqtSlot(str)
    def on_distr_currentTextChanged(self, p0):
#        self.distrx = 'uniform'
        self.distrx = p0
        print(self.distrx)
    
    @pyqtSlot(str)
    def on_fmax_textChanged(self, p0):
        self.fmaxx = self.str2floatx(p0)*1e12
    
    @pyqtSlot(str)
    def on_fmin_textChanged(self, p0):
        self.fminx = self.str2floatx(p0)*1e9
    
    @pyqtSlot(str)
    def on_xi_textChanged(self, p0):
       self.xix = self.str2floatx(p0)      #在波长下切片的数目
    
    @pyqtSlot(str)
    def on_snap_textChanged(self, p0):
        self.snapx = self.str2floatx(p0) # 计算中取样的数目
        self.horizontalSlider.setMaximum(int(self.snapx-1))
        self.horizontalSlider.setMinimum(0)
        self.horizontalSlider.setSingleStep(1)
        self.horizontalSlider.setTickInterval(1)
        self.horizontalSlider_2.setMaximum(int(self.snapx-1))
        self.horizontalSlider_2.setMinimum(0)
        self.horizontalSlider_2.setSingleStep(1)
        self.horizontalSlider_2.setTickInterval(1)
        self.position1 = int(self.snapx-1)
        self.position2 = int(self.snapx -1)
    
    @pyqtSlot(str)
    def on_probez_textChanged(self, p0):
        self.probezx = self.str2floatx(p0)*1e-3
    
#==================================第四页====================================================
    @pyqtSlot(str)
    def on_transversfield_currentTextChanged(self, p0):
        self.Tfield = p0 # 显示横向的选项，有三个
#        print(self.Tfield)
    
    @pyqtSlot(str)
    def on_comboBox_currentTextChanged(self, p0):
        self.farfieldx = p0 #显示远场的选项，两个
#        print(self.Tfield)
    
    @pyqtSlot(int)
    def on_harmonic3_valueChanged(self, p0):
        self.harmonic3x = p0
    
    @pyqtSlot(bool)
    def on_calculateT_clicked(self, checked):
        self.rdist,  self.mat = distf.multimode_field(self.paramters)
    
    @pyqtSlot(bool)
    def on_calculateF_clicked(self, checked):
        Nplain = len(self.rdist)
        # 生成角度分布
        thetaplain0 = np.linspace(0, pi/2, Nplain, endpoint=False)
        self.thetaplainx = difm.mirror_expansion(thetaplain0/pi*180, -1)
        # 计算传输矩阵，1表示为传播距离为1m, 0表示源没有发生偏心
        p2pe = difm.field_matrix('c2s+etheta', self.fre[self.harmonic3x], self.rdist, thetaplain0, 1, 0)
        p2ph = difm.field_matrix('c2s+hphi', self.fre[self.harmonic3x], self.rdist, thetaplain0, 1, 0)
        # 计算远场分布
        far_e = difm.multi_matrix(self.mat[self.harmonic3x], p2pe, 3)
        far_h = difm.multi_matrix(self.mat[self.harmonic3x], p2ph, 3)
        # 将场按照对称结构扩展为-90~90
        self.farfielde = difm.mirror_expansion(np.abs(far_e), 1)
        self.farfieldh = difm.mirror_expansion(np.abs(far_h), 1)
        # 用Lcd显示计算完成
        self.lcdNumber.display(666)
        QApplication.processEvents()
    
    @pyqtSlot(bool)
    def on_Trans_plot_clicked(self, checked):
        if self.Tfield == 'Ez':
            self.tfieldshow.mpl.plot_tfield(self.rdist*1e3,  self.mat[self.harmonic3x][1], self.Tfield+'(a.u.)', self.rax*1e3,  self.rbx*1e3,  self.rex*1e3)
        elif self.Tfield == 'Er':
            self.tfieldshow.mpl.plot_tfield(self.rdist*1e3,  self.mat[self.harmonic3x][2], self.Tfield+'(a.u.)', self.rax*1e3,  self.rbx*1e3,  self.rex*1e3)
        elif self.Tfield == 'Bt':
            self.tfieldshow.mpl.plot_tfield(self.rdist*1e3,  self.mat[self.harmonic3x][0], self.Tfield+'(a.u.)', self.rax*1e3,  self.rbx*1e3,  self.rex*1e3)
        else:
            print('error')
    
    @pyqtSlot(bool)
    def on_Far_plot_clicked(self, checked):
        if self.farfieldx == 'E_theta':
            self.farfield.mpl.plot_field(self.thetaplainx,  self.farfielde, 'degree', 'E')
        elif self.farfieldx == 'H_phi':
            self.farfield.mpl.plot_field(self.thetaplainx, self.farfieldh,  'degree', 'H')
        else:
            print('error!')
    
#===============================第三页===========================================================
    @pyqtSlot(int)
    def on_horizontalSlider_2_valueChanged(self, value):
        self. position2 = value
#        print(self.position2)
    
    @pyqtSlot(bool)
    def on_run2_clicked(self, checked):
        Nele                         = fm2en(self.fmaxx,  self.sigtx)
        dt, self.dz, self.dxi     = slice_gen(self.fre[-1], self.beta, self.vg, self.xix)
        self.micro_q, ele_group   = ele_g(self.sigzx, self.sigrx, self.Qx, self.gama, Nele, self.distzx, self.distrx)
        field_num,field_group= field_g(self.sigzx, self.dxi)
        
        run_num  = int(self.Lx//self.dz)+1
        snap_shot = run_num//self.snapx
        self.power2d     = np.zeros((self.harmonics, run_num))
        self.probe2d      = np.zeros((2, run_num))
        self.field2d        = []
        self.fieldsnap2d = []
        self.electron2d  = []
        for i in range(run_num):
            ele_group, field_group, powerx = e_step(ele_group, field_group, self.params_guide, self.beta, self.dz, dt, self.dxi)
            ele_group, field_group = field_step_2D(ele_group, field_group, self.params_guide, self.beta, self.micro_q, self.dz, dt, self.dxi)
            self.power2d[:, i] = powerx
            self.probe2d[0][i] = dt*i
            self.probe2d[1][i] = np.real(probe_g(self.probezx, field_group, self.dxi))
            # 这里可以按照等间距的时间间隔，将场和电子文件保存下来
            if i % snap_shot == 0:
                self.field2d.append(deepcopy(field_group))                   #将深度复制下的数值加入列表中
                x = field_add(self.field2d[-1],  self.dz,  self.dxi)
                self.electron2d.append(deepcopy(ele_group))                # 必须深度复制，不然数据会变化
                self.fieldsnap2d.append(deepcopy(x))
                self.progressBar_2.setValue(int(i/(run_num-1)*100))    # 赋值到对应的进度条当中
                QApplication.processEvents()                                         #刷新一下进度条
                print(i//snap_shot)
            else:
                pass
        self.progressBar_2.setValue(int(i/(run_num-1)*100))    # 赋值到对应的进度条当中
        QApplication.processEvents()
    
    @pyqtSlot(bool)
    def on_E2d_show_clicked(self, checked):
        if self.electron_type2d == 'Energy':
            self.electronshow2d.mpl.plot_electron(self.electron2d[self.position2][0], self.electron2d[self.position2][1], 'z (mm)',  '$gamma$')
        elif self.electron_type2d == 'R_position':
            self.electronshow2d.mpl.plot_electron(self.electron2d[self.position2][0], np.real(self.electron2d[self.position2][2]),  'z (mm)',  'R (mm)')
        elif self.electron_type2d == 'R-dR':
            self.electronshow2d.mpl.plot_electron(self.electron2d[self.position2][2], np.real(self.electron2d[self.position2][3]),  'R (mm)',  'dR')
        elif self.electron_type2d == 'Current':
            self.electronshow2d.mpl.plot_current(self.electron2d[self.position2][0], self.micro_q, self.beta)
        else:
            print('error')
    
    @pyqtSlot(str)
    def on_field2d_currentTextChanged(self, p0):
        self.field_type2d = p0
    
    @pyqtSlot(str)
    def on_electron2d_currentTextChanged(self, p0):
        self.electron_type2d = p0
    
    @pyqtSlot(int)
    def on_harmonic2_valueChanged(self, p0):
        self.harmonic2x = p0
    
    @pyqtSlot(bool)
    def on_F2d_show_clicked(self, checked):
        if self.field_type2d == 'snapshot_single':
            self.fieldshow2d.mpl.plot_field(self.field2d[self.position2][self.harmonic2x][0], np.real(self.field2d[self.position2][self.harmonic2x][1]), 'z(mm)',  self.field_type2d+'(V/m)')
        elif self.field_type2d == 'power':
            self.fieldshow2d.mpl.plot_single(self.power2d[self.harmonic2x], self.field_type2d+'(MW)')
        elif self.field_type2d == 'probe':
            self.fieldshow2d.mpl.plot_field(self.probe2d[0], self.probe2d[1], 't(s)',  self.field_type2d+'(V/m)')
        elif self.field_type2d == 'snapshot_all':
            self.fieldshow2d.mpl.plot_field(self.fieldsnap2d[self.position2][0], self.fieldsnap2d[self.position2][1], 'z(mm)', self.field_type2d+'(V/m)')
        elif self.field_type2d =='probe_spec':
            self.fieldshow2d.mpl.plot_spec(self.probe2d[1], self.dz/c/self.beta)
        elif self.field_type2d == 'snapshota_spec':
            self.fieldshow2d.mpl.plot_spec(self.fieldsnap2d[self.position2][1], self.dz/c/self.beta)
        else:
            print('error')


#=============================================第二页=================================================
    @pyqtSlot(bool)
    def on_Run1_clicked(self, checked):
        Nele                         = fm2en(self.fmaxx,  self.sigtx)
        dt, self.dz, self.dxi                 = slice_gen(self.fre[-1], self.beta, self.vg, self.xix)
        micro_q, ele_group   = ele_g(self.sigzx, self.sigrx, self.Qx, self.gama, Nele, self.distzx, self.distrx)
        field_num,field_group= field_g(self.sigzx, self.dxi)
        
        run_num  = int(self.Lx//self.dz)+1
        snap_shot = run_num//self.snapx
        self.power1d     = np.zeros((self.harmonics, run_num))
        self.probe1d      = np.zeros((2, run_num))
        self.field1d        = []
        self.fieldsnap1d = []
        self.electron1d  = []
        for i in range(run_num):
            ele_group, field_group, powerx = e_step(ele_group, field_group, self.params_guide, self.beta, self.dz, dt, self.dxi)
            ele_group, field_group = field_step_1D(ele_group, field_group, self.params_guide, self.beta, micro_q, self.dz, dt, self.dxi)
            self.power1d[:, i] = powerx
            self.probe1d[0][i] = dt*i
            self.probe1d[1][i] = np.real(probe_g(self.probezx, field_group, self.dxi))
            # 这里可以按照等间距的时间间隔，将场和电子文件保存下来
            if i % snap_shot == 0:
                self.field1d.append(deepcopy(field_group))
                x = field_add(self.field1d[-1],  self.dz,  self.dxi)
                self.electron1d.append(deepcopy(ele_group))
                self.fieldsnap1d.append(deepcopy(x))
                self.progressBar.setValue(int(i/(run_num-1)*100))    # 赋值到对应的进度条当中
                QApplication.processEvents()
                print(i//snap_shot)
#                print(self.electron1d[0])
            else:
                pass
        self.progressBar.setValue(int(i/(run_num-1)*100))    # 赋值到对应的进度条当中
        QApplication.processEvents()

    @pyqtSlot(int)
    def on_horizontalSlider_valueChanged(self, value):
        self.position1 = value
#        print(self.position1)
        # 图像显示所对应的snap位置，相关的参数在snap的函数当中已经设置好了

    @pyqtSlot(bool)
    def on_E1d_show_clicked(self, checked):
        self.phasespace1d.mpl.plot_electron(self.electron1d[self.position1][0], self.electron1d[self.position1][1], 'z (mm)',  '$gamma$')
    
    @pyqtSlot(bool)
    def on_F1d_show_clicked(self, checked):
        if self.field_type1d == 'snapshot_single':
            self.fieldshow1d.mpl.plot_field(self.field1d[self.position1][self.harmonic1x][0], np.real(self.field1d[self.position1][self.harmonic1x][1]), 'z(mm)',  self.field_type1d)
        elif self.field_type1d == 'power':
            self.fieldshow1d.mpl.plot_single(self.power1d[self.harmonic1x],  self.field_type1d+'(MW)')
        elif self.field_type1d == 'probe':
            self.fieldshow1d.mpl.plot_field(self.probe1d[0], np.real(self.probe1d[1]),'t(s)',  self.field_type1d+'(V/m)')
        elif self.field_type1d == 'snapshot_all':
            self.fieldshow1d.mpl.plot_field(self.fieldsnap1d[self.position1][0], self.fieldsnap1d[self.position1][1], 'z (m)', self.field_type1d)
        elif self.field_type1d =='probe_spec':
            self.fieldshow2d.mpl.plot_spec(self.probe2d[1], self.dz/c/self.beta)
        elif self.field_type1d == 'snapshota_spec':
            self.fieldshow2d.mpl.plot_spec(self.fieldsnap1d[self.position1][1], self.dz/c/self.beta)
        else:
            print('error')
    
    @pyqtSlot(str)
    def on_field1d_currentTextChanged(self, p0):
        self.field_type1d = p0
    
    @pyqtSlot(int)
    def on_harmonic1_valueChanged(self, p0):
        self.harmonic1x = p0
        print(self.harmonic1x)
    
#===================================第一页========================================================
    @pyqtSlot(bool)
    def on_shceme_show_clicked(self, checked):
        self.scheme.setVisible(True)
        self.scheme.mpl.plot_scheme(self.rax*1e3, self.rbx*1e3, self.rex*1e3, self.Lx*1e3, self.sigzx*1e3)
        # 预览图显示
    
    @pyqtSlot(bool)
    def on_para_cal_clicked(self, checked):
        self.paramters=[self.rax, self.rbx, self.rex, self.energyx, self.eps1x, self.eps2x, self.fmaxx, int(self.fmaxx/self.fminx)]
        self.fre, self.kappa, self.vg, self.a0, self.ac = distf.multimode_guide(self.paramters)
        self.params_guide = [self.fre,  self.kappa,  self.vg,  self.ac,  self.Lx]
        self.harmonics      = len(self.fre)
        # 计算所需要的波导参数
    
    @pyqtSlot(bool)
    def on_para_plot_clicked(self, checked):
        self.params.mpl.plot_factor(self.fre, self.kappa, self.vg)
        # 在图中显示波导参数
    
    @pyqtSlot(bool)
    def on_table_show_clicked(self, checked):
        datax = {'Frequency(Hz)':self.fre, 
        'Loss_factor(V/m/nC)':self.kappa, 'Group_velocity':self.vg, 
        'Lossc':self.ac}
        self.df = DataFrame(datax)
        self.model.setDataFrame(self.df)
        # 在表中显示波导参数

if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    ui = MainWindow()
    ui.show()
    sys.exit(app.exec_())
