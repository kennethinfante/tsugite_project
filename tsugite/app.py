#!/usr/bin/env python3

import sys
import os
import time
from math import tan, pi

import numpy as np
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.uic import *
from PyQt5.QtOpenGL import *

from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *

from joint_types import Types
from geometries import Geometries
from view_settings import ViewSettings
from display import Display
from main_window import get_untitled_filename, mainWindow

class MovieSplashScreen(QSplashScreen):

    def __init__(self, movie, parent = None):

        movie.jumpToFrame(0)
        pixmap = QPixmap(movie.frameRect().size())
   
        QSplashScreen.__init__(self, pixmap)
        self.movie = movie
        self.movie.frameChanged.connect(self.repaint)
    
    def showEvent(self, event):
        self.movie.start()
    
    def hideEvent(self, event):
        self.movie.stop()
    
    def paintEvent(self, event):
    
        painter = QPainter(self)
        pixmap = self.movie.currentPixmap()
        self.setMask(pixmap.mask())
        painter.drawPixmap(0, 0, pixmap)

    def sizeHint(self):
        return self.movie.scaledSize()
  

#deal with dpi
QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True) #enable highdpi scaling
QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True) #use highdpi icons
app = QApplication(sys.argv)
movie = QMovie("images/tsugite_loading_3d.gif")

splash = MovieSplashScreen(movie)

splash.show()

start = time.time()

while movie.state() == QMovie.Running and time.time() < start + 1:
    app.processEvents()
#screen = app.screens()[0]
#dpi = screen.physicalDotsPerInch()

window = mainWindow()
window.show()
splash.finish(window)
sys.exit(app.exec_())
