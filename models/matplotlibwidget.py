# from PyQt5 import QtCore
from PyQt5.QtWidgets import QApplication,  QVBoxLayout, QSizePolicy, QWidget #,QMainWindow
import sys
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import numpy as np
import matplotlib.pyplot as plt

class MyMplCanvas(FigureCanvas):
    ''' 创建新的画布 '''
    def __init__(self, parent=None, width = 8,  height= 4,  dpi=100):
        self.fig = plt.figure(figsize=(width, height), dpi=dpi) # new fig
        self.axes = self.fig.add_subplot(111)                     # 增加子图
        #self.axes.hold(False)                                            # 不会叠加在原来的图像上
        super(MyMplCanvas, self).__init__(self.fig)
        FigureCanvas.__init__(self, self.fig)            #
        self.setParent(parent)
        # 可以让图像尽可能地扩展地边沿
        FigureCanvas.setSizePolicy(self, QSizePolicy.Expanding, QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
    
    def plot_factor(self,  array1,  array2, array3,):
#        self.axes = self.fig.add_subplot(111)                     # 增加子图
        self.axes.cla()
        self.axes.plot(array1/1e12,  array2/1e6, 'o-', color = 'r')
        self.axes1=self.axes.twinx()
        self.axes1.cla()
        self.axes1.plot(array1/1e12,  array3, '*-', color = 'b')
#        for i in range(len(array1)):
#            self.axes.text(array1[i], array2[i], '(%.3f, %.3f)'%(array1[i]/1e12, array2[i]/1e6), color='r')
#            self.axes1.text(array1[i], array3[i], '(%.3f, %.3f)'%(array1[i]/1e12, array3[i]), color='b')
        self.axes.set_xlabel('Frequency(THz)')
        self.axes.set_ylabel('Loss Factor(MV/m/nC)')
        self.axes1.set_ylabel('Group Velocity')
        self.draw()
        
    def plot_scheme(self, ra, rb, rex, length,  lengthe):
#        self.axes = self.fig.add_subplot(111)                     # 增加子图
        self.axes.cla()
        plt.xlim(0,  length)
        rect1 = plt.Rectangle((0, ra), length,  rb-ra, color='b', alpha =0.8) # dielectric layer 1
        rect2 = plt.Rectangle((0, rb), length, rex-rb, color='b', alpha =0.3)#dielectric layer 2
        rect3 =plt.Rectangle((0, rex), length, ra/2, color='k', alpha =0.3) # conductor layer
        bunch= plt.Rectangle((length- lengthe,  0), lengthe,  ra/3, color='r', alpha =0.5)  # bunch
        line1  = np.linspace(0,  length,  100)
        line2 = (np.sin(line1/line1[-1]*20)+1)*ra/6
        self.axes.plot(line1,  line2)
        self.axes.add_patch(rect1)
        self.axes.add_patch(rect2)
        self.axes.add_patch(rect3)
        self.axes.add_patch(bunch)
        self.axes.text(length/3, 0, 'Vaccum',  color = 'k')
        self.axes.text(length/3, ra, 'eps1',  color = 'k')
        self.axes.text(length/2, rb,  'eps2',  color = 'k')
        self.axes.text(length/3, rex, 'Conductot',  color = 'k')
        self.axes.set_xlabel('L(mm)')
        self.axes.set_ylabel('R(mm)')
        self.axes.autoscale_view()
        self.draw()
        
    def plot_electron(self,  x_line,  y_line,  x_str,  y_str):
#        self.axes.add_subplot(111)
        self.axes.cla()
        self.axes.plot(x_line, y_line, 'ro')
        self.axes.set_xlabel(x_str)
        self.axes.set_ylabel(y_str)
        self.axes.autoscale_view()
        self.draw()
    
    def plot_field(self, x_line, y_line, x_str, y_str):
#        self.axes.add_subplot(111)
        self.axes.cla()
        self.axes.plot(x_line, y_line)
        self.axes.set_xlabel(x_str)
        self.axes.set_ylabel(y_str)
        self.axes.autoscale_view()
        self.draw()
        
    def plot_single(self, linex, y_str):
        self.axes.cla()
        self.axes.plot(linex)
        self.axes.set_xlabel('step')
        self.axes.set_ylabel(y_str)
        self.axes.autoscale_view()
        self.draw()
    
    def plot_tfield(self, linex, liney, y_str, ra,  rb,  rex):
        self.axes.cla()
        self.axes.plot(linex,  liney/np.max(liney),  color = 'r')
        ymin = np.min(liney)/np.max(liney)
        h = np.max(liney)-np.min(liney)
        rect1 = plt.Rectangle((ra, ymin), rb-ra, h,  color='b', alpha =0.4) # dielectric layer 1
        rect2 = plt.Rectangle((rb, ymin), rex-rb, h,  color='b', alpha =0.2)#dielectric layer 2
        rect3 =plt.Rectangle((rex, ymin), ra/2,h,  color='k', alpha =0.5) # conductor layer
        self.axes.add_patch(rect1)
        self.axes.add_patch(rect2)
        self.axes.add_patch(rect3)
        self.axes.set_xlabel('R (mm)')
        self.axes.set_ylabel(y_str)
        self.axes.autoscale_view()
        self.draw()
    
    def plot_current(self, linex, micro_q,  beta):
        counts,  binx,  barx = plt.hist(linex,  bins = 256)
        thickx = binx[1]-binx[0]
        binxx = binx[0:256]+thickx/2
        c = 2.998e8
        current = counts*micro_q/thickx*c*beta*1e-9
        current[0]=0
        current[-1]=0
        self.axes.cla()
        self.axes.plot(binxx,  current)
        self.axes.set_xlabel('z (mm)')
        self.axes.set_ylabel('Current (A)')
        self.axes.autoscale_view()
        self.draw()
    
    def plot_spec(self, line_y, dt):
        Fmax = 1/dt
        N = len(line_y)
        yff = np.abs(np.fft.fft(line_y))/N*2
        xff = np.linspace(0, Fmax,  N)/1e12
        self.axes.cla()
        self.axes.plot(xff[0:int(N/2)], yff[0:int(N/2)] )
        self.axes.set_xlabel('Frequency (THz)')
        self.axes.set_ylabel('Amplitude')
        self.axes.autoscale_view()
        self.draw()

class matplotlibWidget(QWidget):
    def __init__(self, parent = None):
        super(matplotlibWidget, self).__init__(parent)
        self.initUi()
    
    def initUi(self):
        self.layout = QVBoxLayout(self)
        self.mpl = MyMplCanvas(self, width = 8, height=4, dpi=100)
        self.mpl_ntb = NavigationToolbar(self.mpl, self) #会用来添加工具栏
        
        self.layout.addWidget(self.mpl)
        self.layout.addWidget(self.mpl_ntb)
    
if __name__ == '__main__':
    app = QApplication(sys.argv)
    ui = matplotlibWidget()
#    ui.mpl.start_static_plot()  # 测试静态图效果
    # ui.mpl.start_dynamic_plot() # 测试动态图效果
    ui.show()
    sys.exit(app.exec_())
