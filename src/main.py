"""
  Author: MidnightStudioOfficial
  License: MIT
  Description: This is the main script that loads and connects everything
"""
import sys
from os.path import join, dirname, realpath

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import QApplication, QSplashScreen, QLabel

from core.ui.ui import MainGUI

if __name__ == '__main__':
    """
    Main entry point of the program.
    This script creates a GUI for the Luna LLM application with a splash screen that shows a loading progress.
    """
    image_path = join(dirname(realpath(__file__)), "Data/assets")

    print('Creating GUI')

    # Create the PyQt application
    app = QApplication(sys.argv)

    print('Creating Luna and training')

    gui = MainGUI(image_path)

    gui.show()
    sys.exit(app.exec())
