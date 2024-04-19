from PyQt6.QtWidgets import (
    QLabel, 
    QFrame,
    QVBoxLayout,
    QHBoxLayout,
)
from qfluentwidgets import (NavigationItemPosition, MessageBox, isDarkTheme, setTheme,
                            LineEdit, FluentWindow, SubtitleLabel, setFont,
                            Theme, setThemeColor, NavigationToolButton, NavigationPanel, PushButton)
from PyQt6.QtWidgets import QApplication, QFrame, QHBoxLayout
from qfluentwidgets import (NavigationItemPosition, FluentWindow,
                            NavigationAvatarWidget, qrouter, SubtitleLabel, setFont, InfoBadge,
                            InfoBadgePosition, FluentBackgroundTheme)
from PyQt6.QtCore import Qt
from PyQt6.QtCore import QSize
from PyQt6.QtGui import QIcon, QFont
from PyQt6.QtWidgets import QApplication, QStackedWidget, QHBoxLayout, QLabel, QWidget, QVBoxLayout
from PyQt6.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QScrollArea,
)
from PyQt6.QtCore import Qt, QPropertyAnimation, QEasingCurve, QTimer
from core.ui.components.big_card import BigCardView


class HomeWidget(QFrame):

    def __init__(self, text: str, parent=None):
        super().__init__(parent=parent)
        #self.label = SubtitleLabel(text, self)
        self.hBoxLayout = QHBoxLayout(self)
        #self.home_window_frame = QFrame(self)
        # Chat window layout (vertical)
        #self.home_window_layout = QVBoxLayout(self.home_window_frame)
        self.homeLabel = QLabel('Home', self)
        self.homeLabel.setAlignment(Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignTop)
        # change the labels size to a title size
        font = QFont()
        font.setFamily("Comic Sans MS")
        font.setPointSize(20)  # Change the point size to your desired size
        font.setBold(True)
        self.homeLabel.setFont(font)
        self.hBoxLayout.addWidget(self.homeLabel, alignment=Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignTop) #, 1, Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignTop)

        #self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        #self.home_window_frame.setLayout(self.home_window_layout)
        #self.hBoxLayout.addWidget(self.home_window_frame, 1, Qt.AlignmentFlag.AlignCenter)
        #self.hBoxLayout.addWidget(self.chat_window_frame, 1, Qt.AlignmentFlag.AlignCenter)
        self.setObjectName(text.replace(' ', '-'))
