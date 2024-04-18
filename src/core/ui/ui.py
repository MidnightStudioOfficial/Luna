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

class OLDGUI(FluentWindow): #QMainWindow #QWidget
    def __init__(self, image_path):
        super().__init__()
        #self.master = master

        # Load images using QPixmap
        self.logo_image = QPixmap(f"{image_path}/ava.jfif").scaled(26, 26)

        self.master = self

        # Set window title and size
        self.setWindowTitle("ChatBot")
        self.setFixedSize(600, 688)

        #self.master.bind('<F12>', lambda _: Debugger(self.master))

        # Create main layout (grid)
        self.main_layout = QGridLayout(self)



        # Main chat window frame
        self.chat_window_frame = QFrame(self)
        self.chat_window_frame.setStyleSheet("background-color: #F8F8F8; border-top-left-radius: 7px; border-top-right-radius: 7px;")
        self.main_layout.addWidget(self.chat_window_frame, 1, 0, 1, 1)

        # Chat window layout (vertical)
        self.chat_window_layout = QVBoxLayout(self.chat_window_frame)

        # Chat history display
        self.chat_history_display = QLabel(self)
        self.chat_history_display.setText("Chat history will be displayed here.")
        self.chat_history_display.setStyleSheet("color: #303030; overflow-y: auto; padding: 10px; font-size: 14px;")
        self.chat_window_layout.addWidget(self.chat_history_display)

        # User input frame
        self.user_input_frame = QHBoxLayout()

        # User input textbox
        self.user_input_textbox = LineEdit(self)
        self.user_input_textbox.setPlaceholderText("Type your message...")
        self.user_input_textbox.setStyleSheet("border-radius: 5px; padding: 10px; border: 1px solid #cccccc; font-size: 14px;")
        self.user_input_frame.addWidget(self.user_input_textbox)

        # Send button
        self.send_button = PushButton("Send", self)
        self.send_button.setFixedSize(QSize(100, 30))
        self.send_button.setStyleSheet("background-color: #20A3DD; color: white; border: 0px; border-radius: 5px; padding: 5px 10px; font-size: 12px;")
        #self.send_button.clicked.connect(self.send_button_event)  # Connect to send button function
        self.user_input_frame.addWidget(self.send_button)

        self.chat_window_layout.addLayout(self.user_input_frame)
        self.chat_window_frame.setLayout(self.chat_window_layout)
        self.main_layout.addWidget(self.chat_window_frame) #, 0, 0, 1, 1

      

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

        # Bind the 'Return' key event to the send_message method
        #self.UserField.bind('<Return>', lambda event: self.MessagesController.send_message(None))

class MainGUI(FluentWindow):

    def __init__(self, image_path):
        super().__init__()

        # create sub interface
        self.homeInterface = Widget('Home', self)
        self.musicInterface = ChatWidget('Chat', self)
        self.videoInterface = Widget('Video Interface', self)
        self.folderInterface = Widget('Folder Interface', self)
        self.settingInterface = Widget('Setting Interface', self)
        self.albumInterface = Widget('Album Interface', self)
        self.albumInterface1 = Widget('Album Interface 1', self)
        self.albumInterface2 = Widget('Album Interface 2', self)
        self.albumInterface1_1 = Widget('Album Interface 1-1', self)

        self.initNavigation()
        self.initWindow()

              

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


    def initNavigation(self):
        self.addSubInterface(self.homeInterface, FIF.HOME, 'Home')
        self.addSubInterface(self.musicInterface, FIF.CHAT, 'Chat')
        #self.addSubInterface(self.videoInterface, FIF.VIDEO, 'Video library')

        self.navigationInterface.addSeparator()

        #self.addSubInterface(self.albumInterface, FIF.ALBUM, 'Albums', NavigationItemPosition.SCROLL)
        #self.addSubInterface(self.albumInterface1, FIF.ALBUM, 'Album 1', parent=self.albumInterface)
        #self.addSubInterface(self.albumInterface1_1, FIF.ALBUM, 'Album 1.1', parent=self.albumInterface1)
        #self.addSubInterface(self.albumInterface2, FIF.ALBUM, 'Album 2', parent=self.albumInterface)
        #self.addSubInterface(self.folderInterface, FIF.FOLDER, 'Folder library', NavigationItemPosition.SCROLL)

        # add custom widget to bottom
        # self.navigationInterface.addWidget(
        #     routeKey='avatar',
        #     widget=NavigationAvatarWidget('zhiyiYo', 'resource/shoko.png'),
        #     onClick=self.showMessageBox,
        #     position=NavigationItemPosition.BOTTOM,
        # )

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
        self.setWindowTitle('PyQt-Fluent-Widgets')

        desktop = QApplication.screens()[0].availableGeometry()
        w, h = desktop.width(), desktop.height()
        self.move(w//2 - self.width()//2, h//2 - self.height()//2)

        # use custom background color theme (only available when the mica effect is disabled)
        self.setCustomBackgroundColor(*FluentBackgroundTheme.DEFAULT_BLUE)
        # self.setMicaEffectEnabled(False)

        # set the minimum window width that allows the navigation panel to be expanded
        # self.navigationInterface.setMinimumExpandWidth(900)
        # self.navigationInterface.expand(useAni=False)


