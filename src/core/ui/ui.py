from os.path import join
import logging
from threading import Thread


from core.ui.skills.skills_page import SkillGUI
from core.controllers.profile.profile import Profile
from core.base.debug import DebugGUI
from core.ui.widgets.ctk_xyframe import CTkXYFrame
from core.ui.utils.win_style import *
from core.ui.weather.weather import WeatherGUI
from core.base.global_vars import *
from core.voice.wake_word import WakeWord
from core.ui.wakeword.wakeword3 import WakeWordGUI
from core.controllers.messages.messages import MessagesController
from core.TkDeb.TkDeb import Debugger

from .Pages.chat import ChatWidget
from .Pages.home import HomeWidget

from core.engine.EngineCore import engine

from PyQt6.QtWidgets import (
    QApplication,
    QWidget,
    QMainWindow,
    QPushButton,
    QLabel,
    QFrame,
    QVBoxLayout,
    QHBoxLayout,
    QStackedWidget,
    QComboBox,
    QLineEdit
)
from PyQt6.QtGui import QIcon, QPixmap  # For images and icons
from PyQt6.QtCore import Qt  # For alignment and flags
from PyQt6.QtGui import QPixmap, QFont, QBrush, QPalette
from PyQt6.QtWidgets import QGridLayout
from PyQt6.QtCore import QSize

import sys
from PyQt6.QtCore import Qt, QEvent
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QApplication, QStackedWidget, QHBoxLayout, QLabel, QWidget, QVBoxLayout

from qfluentwidgets import (NavigationItemPosition, MessageBox, isDarkTheme, setTheme,
                            LineEdit, FluentWindow, SubtitleLabel, setFont,
                            Theme, setThemeColor, NavigationToolButton, NavigationPanel, PushButton)
from PyQt6.QtWidgets import QApplication, QFrame, QHBoxLayout
from qfluentwidgets import (NavigationItemPosition, FluentWindow,
                            NavigationAvatarWidget, qrouter, SubtitleLabel, setFont, InfoBadge,
                            InfoBadgePosition, FluentBackgroundTheme)
from qfluentwidgets import FluentIcon as FIF


from qframelesswindow import FramelessWindow, StandardTitleBar

DEBUG_GUI = True

if DEBUG_CHATBOT is None or DEBUG_CHATBOT is True: 
    if DEBUG_GUI is False:
        print('Importing pyttsx3')
        from pyttsx3 import init as pyttsx3_init

        print("Importing chatbot")
        from core.chatbot.chatbot import Chatbot

print("Importing DONE")

logging.basicConfig(level=logging.INFO)

# Disable debug messages from comtypes
logging.getLogger("comtypes").setLevel(logging.INFO)


# Dictionary containing image data
image_data = {
    "logo_image": {"name": "ava.jfif", "size": (26, 26)},
    "large_test_image": {"name": "Welcome.png", "size": (290, 118)},
    "image_icon_image": {"name": "home.png", "size": (20, 20)},
    "image_weather_icon_image": {"name": "weather.png", "size": (20, 20)},
    "image_news_icon_image": {"name": "news.png", "size": (20, 20)},
    "image_bell_icon_image": {"name": "bell.png", "size": (20, 20)},
    "image_fire_icon_image": {"name": "fire.png", "size": (20, 20)}
}


class Widget(QFrame):

    def __init__(self, text: str, parent=None):
        super().__init__(parent=parent)
        self.label = SubtitleLabel(text, self)
        self.hBoxLayout = QHBoxLayout(self)

        setFont(self.label, 24)
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.hBoxLayout.addWidget(self.label, 1, Qt.AlignmentFlag.AlignCenter)
        self.setObjectName(text.replace(' ', '-'))


class MainGUI(FluentWindow):

    def __init__(self, image_path):
        super().__init__()

        setTheme(Theme.DARK)

        # create sub interface
        self.homeInterface = HomeWidget('Home', self)
        self.chatInterface = ChatWidget('Chat', self)
        self.createInterface = Widget('Create', self)
        self.profileInterface = Widget('Profile', self)
        self.skillsInterface = Widget('Skills', self)
        self.moreInterface = Widget('More', self)
        self.settingInterface = Widget('Settings', self)

        self.initNavigation()
        self.initWindow()

        #setTheme(theme=Theme.DARK, lazy=True)

        if DEBUG_CHATBOT is None or DEBUG_CHATBOT is True:
            if DEBUG_GUI is False:
                # Set up voice
                self.engine = pyttsx3_init()
                self.voices = self.engine.getProperty('voices')
                self.engine.setProperty('voice', self.voices[1].id)  # Index 1 for female voice
                self.engine.setProperty('rate', 150)  # Adjust rate to 150 words per minute
                self.engine.setProperty('volume', 0.7)  # Adjust volume to 70% of maximum
                self.engine.setProperty('pitch', 110)  # Adjust pitch to 110% of default
                del self.voices

        self.recognize_thread = None
        self.stoped_lisening = False
        self.message_count = 0
        self.is_lisening_wakeword = False

        if DEBUG_CHATBOT is None or DEBUG_CHATBOT is True:
            if DEBUG_GUI is False:
                # Initialize chatbot
                self.chatbot = engine.MainEngine()
                #self.chatbot.train_bot()  # Train the chatbot

        if DEBUG_CHATBOT is None or DEBUG_CHATBOT is True:
            if DEBUG_GUI is False:
                self.MessagesController = MessagesController(
                    voice_engine=self.engine,
                    chat_frame=self.chat_frame,
                    UserField=self.UserField,
                    AITaskStatusLbl=self.AITaskStatusLbl,
                    current_chat_bubble=self.current_chat_bubble,
                    logo_image=self.logo_image,
                    chatbot=self.chatbot
                )
        else:
            if DEBUG_GUI is False:
                self.MessagesController = MessagesController(
                    chat_frame=self.chat_frame,
                    UserField=self.UserField,
                    AITaskStatusLbl=self.AITaskStatusLbl,
                    current_chat_bubble=self.current_chat_bubble,
                    logo_image=self.logo_image
                )

    def setQss(self):
        color = 'dark' # if isDarkTheme() else 'light'
        with open(f'Data/{color}/demo.qss', encoding='utf-8') as f:
            self.setStyleSheet(f.read())

    def initNavigation(self):
        self.addSubInterface(self.homeInterface, FIF.HOME, 'Home')
        self.addSubInterface(self.chatInterface, FIF.CHAT, 'Chat')
        self.addSubInterface(self.createInterface, FIF.EDIT, 'Create')
        #self.addSubInterface(self.profileInterface, FIF.EDIT, 'Profile')
        self.addSubInterface(self.skillsInterface, FIF.BOOK_SHELF, 'Skills')
        self.addSubInterface(self.moreInterface, FIF.MORE, '"More')

        self.navigationInterface.addSeparator()

        # add custom widget to bottom
        self.navigationInterface.addWidget(
            routeKey='avatar',
            widget=NavigationAvatarWidget('zhiyiYo', 'Data/Icons/user.png'),
            #onClick=self.showMessageBox,
            position=NavigationItemPosition.BOTTOM,
        )

        self.addSubInterface(self.settingInterface, FIF.SETTING, 'Settings', NavigationItemPosition.BOTTOM)

        # add badge to navigation item
        #item = self.navigationInterface.widget(self.videoInterface.objectName())
        #InfoBadge.attension(
        #    text=9,
        #    parent=item.parent(),
        #    target=item,
        #    position=InfoBadgePosition.NAVIGATION_ITEM
        #)

        # NOTE: enable acrylic effect
        self.navigationInterface.setAcrylicEnabled(True)

    def initWindow(self):
        self.resize(900, 700)
        #self.setWindowIcon(QIcon(':/qfluentwidgets/images/logo.png'))
        self.setWindowTitle('Luna')

        desktop = QApplication.screens()[0].availableGeometry()
        w, h = desktop.width(), desktop.height()
        self.move(w//2 - self.width()//2, h//2 - self.height()//2)

        # use custom background color theme (only available when the mica effect is disabled)
        self.setCustomBackgroundColor(*FluentBackgroundTheme.DEFAULT_BLUE) #*FluentBackgroundTheme.DEFAULT_BLUE
        # self.setMicaEffectEnabled(False)

        # set the minimum window width that allows the navigation panel to be expanded
        # self.navigationInterface.setMinimumExpandWidth(900)
        # self.navigationInterface.expand(useAni=False)

        #self.setQss()
