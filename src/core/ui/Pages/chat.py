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
from PyQt6.QtGui import QIcon
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
from qfluentwidgets import SmoothScrollArea

class BubbleLabel(QLabel):
    def __init__(self, text, parent=None, is_user=False):
        super().__init__(text, parent)

        self.setTextFormat(Qt.TextFormat.RichText)
        self.setAlignment(Qt.AlignmentFlag.AlignLeft if not is_user else Qt.AlignmentFlag.AlignRight)
        self.setWordWrap(True)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
        
        # Set styles for user and bot bubbles
        if is_user:
            self.setStyleSheet("""
                background-color: #DCF8C6;
                border-radius: 10px;
                padding: 8px 12px;
                margin-bottom: 5px;
            """)
        else:
            self.setStyleSheet("""
                background-color: #FFFFFF;
                border-radius: 10px;
                padding: 8px 12px;
                margin-bottom: 5px;
            """)

    def _update_geometry(self):
        # Adjust bubble size to fit text
        fm = self.fontMetrics()
        text_width = fm.boundingRect(self.text()).width()
        self.setFixedWidth(text_width + 24)  # Add padding

class ChatWindow(QFrame):
    def __init__(self): #, parent=None
        super().__init__()
        self.setWindowTitle("Chat Box")

        # Main layout
        #main_layout = parent #QVBoxLayout()
        main_layout = QHBoxLayout(self)
        self.setLayout(main_layout)

        # Scrolling area for messages
        self.scroll_area = SmoothScrollArea()
        self.scroll_area.setWidgetResizable(True)
        main_layout.addWidget(self.scroll_area)

        # Widget to hold chat bubbles
        self.chat_widget = QWidget()
        self.chat_layout = QVBoxLayout()
        self.chat_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.chat_widget.setLayout(self.chat_layout)
        self.scroll_area.setWidget(self.chat_widget)

        # Input area
        #self.input_box = QLineEdit()
        #self.send_button = QPushButton("Send")
        #self.send_button.clicked.connect(self.send_message)

        #input_layout = QHBoxLayout()
        #input_layout.addWidget(self.input_box)
        #input_layout.addWidget(self.send_button)

        #main_layout.addLayout(input_layout)

    def send_message(self, text):
        user_text = text#self.input_box.text()
        if user_text:
            self.add_message(user_text, is_user=True)
            #self.input_box.clear()

            # Add bot response (example)
            self.add_message("Bot: You said: " + user_text) 

    def add_message(self, text, is_user=False):
        bubble = BubbleLabel(text, is_user=is_user)
        self.chat_layout.addWidget(bubble)

        # Slide-in animation
        animation = QPropertyAnimation(bubble, b"geometry") #geometry
        animation.setDuration(400)
        animation.setStartValue(bubble.geometry().translated(-bubble.width(), 0))
        animation.setEndValue(bubble.geometry())
        animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        animation.start()

        # Scroll to bottom
        self.scroll_area.verticalScrollBar().setValue(self.scroll_area.verticalScrollBar().maximum())

class ChatWidget(QFrame):

    def __init__(self, text: str, parent=None):
        super().__init__(parent=parent)
        #self.label = SubtitleLabel(text, self)
        self.hBoxLayout = QHBoxLayout(self)
        # Main chat window frame
        self.chat_window_frame = QFrame(self)
        self.chat_window_frame.setStyleSheet("background-color: #F8F8F8; border-top-left-radius: 7px; border-top-right-radius: 7px;")
        self.hBoxLayout.addWidget(self.chat_window_frame) #, 1, 0, 1, 1

        # Chat window layout (vertical)
        self.chat_window_layout = QVBoxLayout(self.chat_window_frame)

        # Chat history display
        self.chat_history_display = ChatWindow() #QLabel(self)
        self.chat_history_display.setFixedSize(600, 688)
        #self.chat_history_display.setText("Chat history will be displayed here.")
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
        self.send_button.clicked.connect(lambda: self.chat_history_display.send_message(self.user_input_textbox.text()))  # Connect to send button function
        self.user_input_frame.addWidget(self.send_button)

        self.chat_window_layout.addLayout(self.user_input_frame)
        self.chat_window_frame.setLayout(self.chat_window_layout)
        #self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.hBoxLayout.addWidget(self.chat_window_frame, 1, Qt.AlignmentFlag.AlignCenter)
        self.setObjectName(text.replace(' ', '-'))
