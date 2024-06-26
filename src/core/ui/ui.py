from tkinter import *
import tkinter as tk
import customtkinter as ctk
from os.path import join
import logging
from PIL import Image, ImageTk, ImageDraw
from threading import Thread


from core.ui.skills.skills_page import SkillGUI
from core.ui.music import Sounder
from core.controllers.profile.profile import Profile
from core.base.debug import DebugGUI
from core.ui.widgets.CTkScrollableDropdown.ctk_scrollable_dropdown import CTkScrollableDropdown
from core.ui.widgets.CTkToolTip.ctk_tooltip import CTkToolTip
from core.ui.widgets.ctk_xyframe import CTkXYFrame
from core.ui.utils.win_style import *
from core.ui.weather.weather import WeatherGUI
from core.base.global_vars import *
from core.voice.wake_word import WakeWord
from core.ui.wakeword.wakeword3 import WakeWordGUI
from core.controllers.messages.messages import MessagesController
from core.TkDeb.TkDeb import Debugger

from core.engine.EngineCore import engine

if DEBUG_CHATBOT is None or DEBUG_CHATBOT is True: 
    print('Importing pyttsx3')
    from pyttsx3 import init as pyttsx3_init

    print("Importing chatbot")
    from core.chatbot.chatbot import Chatbot

print("Importing DONE")

logging.basicConfig(level=logging.INFO)

# Disable debug messages from comtypes
logging.getLogger("comtypes").setLevel(logging.INFO)


### SWITCHING BETWEEN FRAMES ###
def raise_frame(frame):
    frame.tkraise()


class ChatBotGUI:
    def __init__(self, master, splash_screen, image_path):
        """
        Initialize the ChatBotGUI object.

        :param master: The master window
        :type master: Tk
        :param splash_screen: The splash screen object
        :type splash_screen: SplashScreen
        """
        # Initialize the ChatBotGUI object
        self.master = master

        # Set the title and geometry of the master window
        master.title("ChatBot")
        master.geometry("600x688")

        self.master.bind('<F12>', lambda _: Debugger(self.master))

        # Prevent the master window from being resized (commented out)
        # master.resizable(False, False)

        # Set the grid layout to 1 row and 2 columns
        master.grid_rowconfigure(0, weight=1)
        master.grid_columnconfigure(1, weight=1)

        print("checking if a profile exists")
        from core.ui.signup.profile_checker import ProfileChecker
        checker = ProfileChecker()
        checker.create_profile(self.master)

        self.profile_controller = Profile()

        print("Importing user_profile")
        from core.ui.user_profile.userprofile import ProfileClass

        self.min_w = 50 # Minimum width of the frame
        self.max_w = 200 # Maximum width of the frame
        self.cur_width = self.min_w # Increasing width of the frame
        self.expanded = False # Check if it is completely exanded

        # Create CTkImage objects for various images
        splash_screen.set_text("Loading Images")  # Set text for splash screen indicating image loading

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

        # Load and assign images to attributes using image data
        for attribute, data in image_data.items():
            image = Image.open(join(image_path, data["name"]))  # Open the image file
            setattr(self, attribute, ctk.CTkImage(image, size=data["size"]))  # Assign the image to an attribute with specified size

        self.home_image = ctk.CTkImage(light_image=Image.open(join(image_path, "home.png")), dark_image=Image.open(join(image_path, "home.png")), size=(20, 20))
        self.chat_image = ctk.CTkImage(light_image=Image.open(join(image_path, "chat.png")), dark_image=Image.open(join(image_path, "chat.png")), size=(20, 20))
        self.add_user_image = ctk.CTkImage(light_image=Image.open(join(image_path, "settings.png")), dark_image=Image.open(join(image_path, "settings.png")), size=(20, 20))
        self.add_DNA_image = ctk.CTkImage(light_image=Image.open(join(image_path, "DNA.png")), dark_image=Image.open(join(image_path, "DNA.png")), size=(20, 20))
        self.add_profile_image = ctk.CTkImage(Image.open(join(image_path, "profile.png")))
        self.add_profile_image2 = ctk.CTkImage(Image.open(join(image_path, "profile.png")), size=(40, 40))
        self.add_skills_image = ctk.CTkImage(Image.open(join(image_path, "box.png")))
        self.add_music_image = ctk.CTkImage(Image.open(join(image_path, "music.png")))
        self.add_more_image = ctk.CTkImage(Image.open(join(image_path, "more.png")))

        # create navigation frame
        splash_screen.set_text("Creating gui")

        # Create a navigation frame
        self.navigation_frame = ctk.CTkFrame(master, corner_radius=7)
        self.navigation_frame.grid(row=0, column=0, sticky="nsew")
        self.navigation_frame.grid_rowconfigure(8, weight=1) #7

        # Define navigation buttons and their properties
        navigation_buttons = [
            {"attribute": "home_button", "text": "Home", "image": self.home_image, "command": self.home_button_event},
            {"attribute": "frame_2_button", "text": "Chat", "image": self.chat_image, "command": self.frame_2_button_event},
            {"attribute": "frame_3_button", "text": "Settings", "image": self.add_user_image, "command": self.frame_3_button_event},
            {"attribute": "frame_DNA_button", "text": "DNA", "image": self.add_DNA_image, "command": self.frame_DNA_button_event},
            {"attribute": "frame_profile_button", "text": "Profile", "image": self.add_profile_image, "command": self.frame_profile_button_event},
            {"attribute": "frame_skills_button", "text": "Skills", "image": self.add_skills_image, "command": self.frame_skills_button_event},
            {"attribute": "frame_more_button", "text": "More", "image": self.add_more_image, "command": self.frame_more_button_event}
        ]

        # Create the navigation frame label
        self.navigation_frame_label = ctk.CTkLabel(self.navigation_frame, text="  Luna", image=self.logo_image,
                                                compound="left", font=ctk.CTkFont(size=15, weight="bold"))
        self.navigation_frame_label.grid(row=0, column=0, padx=20, pady=20)

        # Create navigation buttons and assign attributes to self
        for index, nav_button in enumerate(navigation_buttons, start=1):
            button = ctk.CTkButton(
                self.navigation_frame,
                corner_radius=0,
                height=40,
                border_spacing=10,
                text=nav_button["text"],
                fg_color="transparent",
                text_color=("gray10", "gray90"),
                hover_color=("gray70", "gray30"),
                image=nav_button["image"],
                anchor="w",
                command=nav_button["command"]
            )
            setattr(self, nav_button["attribute"], button)
            button.grid(row=index, column=0, sticky="ew")

        # Create an appearance mode menu
        self.appearance_mode_menu = ctk.CTkOptionMenu(self.navigation_frame, values=["Dark", "Light", "System"],
                                                    command=self.change_appearance_mode_event)
        self.appearance_mode_menu.grid(row=8, column=0, padx=20, pady=20, sticky="s")

        # create home frame
        splash_screen.set_text("Creating home frame")
        # Creating a custom frame with transparent background to hold the buttons
        self.home_frame = ctk.CTkFrame(master, corner_radius=0, fg_color="transparent")
        self.home_frame.grid(row=0, column=0, sticky="w")
        # Configuring the column in the frame to expand and fill the available space
        self.home_frame.grid_columnconfigure(0, weight=1)

        # Adding the first button for profile selection
        self.profile_button = ctk.CTkButton(self.home_frame, text="", image=self.add_profile_image2, fg_color="transparent", corner_radius=0, width=40, height=40, border_width=0, border_spacing=0, compound="left")
        self.profile_button.grid(row=0, column=0, sticky="nw")

        # Creating a scrollable dropdown menu for profile selection
        CTkScrollableDropdown(self.profile_button, values=self.profile_controller.list_profiles(), height=270, resize=False, button_height=30,
                      scrollbar=True, width=100)

        # Adding a tooltip to show a message when hovering over the profile button
        self.profile_button_tooltip = CTkToolTip(self.profile_button, delay=0.8, message=str(TOOLTIP_MESSAGES["profile_button"]))

        # Adding the second button for mail notifications
        self.mail_button = ctk.CTkButton(self.home_frame, text="", image=self.image_bell_icon_image, fg_color="transparent")
        self.mail_button.grid(row=0, column=0, sticky="ne", padx=5, pady=5)   # Placing the button on the top-right corner with some padding for aesthetics
        self.mail_button.configure(width=30, height=30)
        self.mail_button_tooltip = CTkToolTip(self.mail_button, delay=0.8, message=str(TOOLTIP_MESSAGES["mail_button"]))

        # create welcome image
        self.home_frame_large_image_label = ctk.CTkLabel(self.home_frame, text="", image=self.large_test_image)
        self.home_frame_large_image_label.grid(row=0, column=0, padx=20, pady=10)
        widgets = [
            (ctk.CTkLabel(self.home_frame, text="Welcome back!", font=ctk.CTkFont(family='Lucida Console', size=15, weight="bold")), 1),
            (ctk.CTkButton(self.home_frame, text="Get the Weather", image=self.image_weather_icon_image, compound="right"), 2),
            (ctk.CTkButton(self.home_frame, text="Read the News", image=self.image_news_icon_image, compound="right"), 3),
            (ctk.CTkButton(self.home_frame, text="Get Cozy and Chat", image=self.image_fire_icon_image, compound="right", anchor="w"), 4)
        ]

        for widget, row in widgets:
            widget.grid(row=row, column=0, padx=20, pady=10)

        version_button = ctk.CTkButton(master, text="V0.0.1", width=96, command=self.debug_click)
        version_button.grid(sticky="se", column=1)

        self.current_chat_bubble = False

        # create second frame
        splash_screen.set_text("Creating chat page")
        self.second_frame = ctk.CTkFrame(master, corner_radius=0, fg_color="transparent")

        # Create three frames with different background colors
        self.root1 = ctk.CTkFrame(self.second_frame, bg_color=chatBgColor)
        self.root2 = ctk.CTkFrame(self.second_frame, bg_color=background)
        self.root3 = ctk.CTkFrame(self.second_frame, bg_color=background)

        # Grid all frames to the same cell in the grid layout
        # This will cause the frames to overlap each other
        for f in (self.root1, self.root2, self.root3):
            f.grid(row=0, column=0, sticky='news')

        if self.current_chat_bubble == False:
            self.chat_frame = ctk.CTkTextbox(self.root1, width=380, height=551, fg_color=chatBgColor)
        else:
            self.chat_frame = CTkXYFrame(self.root1, width=380, height=551, fg_color=chatBgColor)
            self.chat_frame.pack_propagate(0)
        self.chat_frame.pack(padx=10)

        # Create a new CTkFrame object with height 100, transparent foreground color and '#dfdfdf' background color
        self.bottomFrame1 = ctk.CTkFrame(self.root1, height=100, fg_color="transparent", bg_color='#dfdfdf')

        # Pack the bottomFrame1 to fill the X direction and be placed at the bottom of its parent widget
        self.bottomFrame1.pack(fill=X, side=BOTTOM)

        # Create a new CTkFrame object with transparent foreground color
        self.VoiceModeFrame = ctk.CTkFrame(self.bottomFrame1, fg_color="transparent")

        # Pack the VoiceModeFrame to fill both X and Y directions
        self.VoiceModeFrame.pack(fill=BOTH)

        # Create a new CTkFrame object with transparent foreground color
        self.TextModeFrame = ctk.CTkFrame(self.bottomFrame1, fg_color="transparent")

        # Pack the TextModeFrame to fill both X and Y directions
        self.TextModeFrame.pack(fill=BOTH)

        self.TextModeFrame.pack_forget()

        # Create PhotoImage objects for the light and dark central button images
        self.cblLightImg = PhotoImage(file="Data/images/centralButton.png")
        self.cblDarkImg = PhotoImage(file="Data/images/centralButton1.png")

        # Set the current central button image to the dark image
        self.cblimage = self.cblDarkImg

        # Create a CTkLabel object with the central button image and transparent foreground color
        self.cbl = ctk.CTkLabel(self.VoiceModeFrame, image=self.cblimage, fg_color="transparent")

        # Pack the central button label with a vertical padding of 17 pixels
        self.cbl.pack(pady=17)

        # Create a CTkLabel object with the text 'Offline', font size 16 and foreground color '#203647'
        self.AITaskStatusLbl = ctk.CTkLabel(self.VoiceModeFrame, text='    Offline', font=('montserrat', 16), fg_color="#203647")

        # Place the AI task status label at position (165, 32) in its parent widget
        self.AITaskStatusLbl.place(x=165, y=32)

        # Keyboard Button
        # Load and resize the light keyboard image
        self.kbphLight = PhotoImage(file="Data/images/keyboard.png")
        self.kbphLight = self.kbphLight.subsample(2, 2)

        # Load and resize the dark keyboard image
        self.kbphDark = PhotoImage(file="Data/images/keyboard1.png")
        self.kbphDark = self.kbphDark.subsample(2, 2)

        # Choose the appropriate keyboard image based on the KCS_IMG value
        if KCS_IMG == 1:
            self.kbphimage = self.kbphDark
        else:
            self.kbphimage = self.kbphLight

        # Create a Tkinter button with the keyboard image
        self.kbBtn = ctk.CTkButton(
            self.VoiceModeFrame,
            text='',
            image=self.kbphimage,
            height=30,
            width=30,
            command=self.changeChatMode,
            fg_color="transparent"
        )
        self.kbBtn.place(x=25, y=30)

        # Mic
        self.micImg = PhotoImage(file="Data/images/mic.png")
        self.micImg = self.micImg.subsample(2, 2)
        self.micBtn = ctk.CTkButton(self.TextModeFrame, text='', image=self.micImg, height=30,
                                    width=30, fg_color="transparent", command=self.changeChatMode)
        self.micBtn.place(relx=1.0, y=30, x=-20, anchor="ne")

        # Text Field
        # Load the text field image
        self.TextFieldImg = PhotoImage(file='Data/images/textField.png')

        # Create a label for the user field with the text field image
        self.UserFieldLBL = ctk.CTkLabel(
            self.TextModeFrame,
            text='',
            image=self.TextFieldImg,
            fg_color="transparent"
        )
        self.UserFieldLBL.pack(pady=17, side=LEFT, padx=10)

        # Create the user field entry widget
        self.UserField = ctk.CTkEntry(
            self.TextModeFrame,
            text_color='white',
            bg_color='#203647',
            font=('Montserrat', 16),
            width=304
        )
        self.UserField.place(x=16, y=30)
        self.UserField.insert(0, "Ask me anything...")

        # Load and resize the image
        image = Image.open("Data/assets/ava.jfif")
        image = image.resize((30, 30))  # Adjust the size as needed

        # Create a circular mask for the button
        mask = Image.new("L", (30, 30), 0)
        draw = ImageDraw.Draw(mask)
        draw.ellipse((0, 0, 30, 30), fill=255)

        # Apply the circular mask to the image
        image.putalpha(mask)

        # Create a PhotoImage from the modified image
        photo = ImageTk.PhotoImage(image)

        self.botBtn = ctk.CTkButton(self.VoiceModeFrame, image=photo, height=30,
                                    width=30, border_width=0, text="", fg_color="transparent", corner_radius=400)
        self.botBtn.place(relx=1.0, y=30, x=-20, anchor="ne")

        raise_frame(self.root1)

        # create third frame
        splash_screen.set_text("Creating settings page")
        self.third_frame = ctk.CTkFrame(master, corner_radius=0, fg_color="transparent")
        # Create a list of frame names and label texts
        frames = [
            ("frame_1", "Chat"),
            ("frame_2", "Theme"),
            ("frame_3", "Other")
        ]

        # Create the main label
        self.settings_frame_large_label = ctk.CTkLabel(self.third_frame, text="Settings", font=ctk.CTkFont(family='Lucida Console', size=15, weight="bold"))
        self.settings_frame_large_label.grid(row=0, column=0, padx=20, pady=10)
        self.settings_frame_large_label.pack(anchor=tk.CENTER)

        # Create frames and labels using loops
        for frame_name, label_text in frames:
            splash_screen.set_text("Creating " + str(frame_name))
            frame = ctk.CTkFrame(self.third_frame)
            frame.pack(pady=20, padx=10, fill='x', expand=True)

            label = ctk.CTkLabel(frame, text=label_text, font=ctk.CTkFont(family='Lucida Console', size=15, weight="bold"))
            label.pack(ipady=4)

            # Assign the created frame to an attribute with the corresponding frame name
            setattr(self, frame_name, frame)

        self.chat_bubble_switch_var = ctk.StringVar(value="off")

        self.chat_bubble_enable = ctk.CTkSwitch(self.frame_1, text="New chat bubble", command=self.chat_bubble_enable_event,
                                 variable=self.chat_bubble_switch_var, onvalue="on", offvalue="off")
        self.chat_bubble_enable.pack()

        # Initialize a StringVar to hold the state of the segmented button
        self.segemented_button_var = ctk.StringVar(value="blue")

        # Create a segmented button widget with three values: "blue", "green", "dark-blue"
        self.segemented_button = ctk.CTkSegmentedButton(self.frame_2, values=["blue", "green", "dark-blue"],
                                                     variable=self.segemented_button_var)
        self.segemented_button.pack()  # Pack the segmented button onto the frame

        # Create a button to reset training data
        self.reset_train_button = ctk.CTkButton(self.frame_1, border_width=0, text="Reset training data")
        self.reset_train_button.pack() # Pack the reset training button onto the frame

        # Create an entry widget to input "Window Style" text
        self.entry = ctk.CTkEntry(self.frame_3, width=240, placeholder_text="Window Style")
        self.entry.pack(fill='x', padx=10, pady=10)  # Pack the entry widget onto the frame

        # Create a scrollable dropdown widget for selecting a window style
        self.style_dropdown = CTkScrollableDropdown(self.entry, values=STYLES_LIST, command=self.style_dropdown_click,
                            autocomplete=True) # Using autocomplete
        self.style_dropdown_tooltip = CTkToolTip(self.entry, delay=0.7, message=str(TOOLTIP_MESSAGES["style_dropdown"]))
        self.entry.insert(0, 'Window Style')  # Insert default text to the entry widget

        label = ctk.CTkLabel(self.frame_1, text="Chat Bubble Corner Radius", font=ctk.CTkFont(family='Sans Serif', size=13, weight="bold"))
        label.pack(ipady=2)
        # Create a slider widget with a range from 0 to 100
        self.slider_1 = ctk.CTkSlider(master=self.frame_1, from_=0, to=100)

        self.slider_1.pack(padx=10)

        # create the DNA frame
        self.DNA_frame = ctk.CTkFrame(master, corner_radius=0, fg_color="transparent")

        self.DNA_frame_large_label = ctk.CTkLabel(self.DNA_frame, text="DNA: Create A new chatbot!", font=ctk.CTkFont(family='Lucida Console', size=15, weight="bold")) #, font=ctk.CTkFont(size=15, weight="bold")
        self.DNA_frame_large_label.grid(row=0, column=0, padx=20, pady=10)
        self.DNA_frame_large_label.place(relx=0.5, rely=0.07, anchor=tk.CENTER)

        self.DNA_label_gender = ctk.CTkLabel(self.DNA_frame, text="Gender")
        self.DNA_label_gender.place(relx=0.5, rely=0.16666, anchor=tk.CENTER)
        self.DNA_combobox_gender = ctk.CTkOptionMenu(self.DNA_frame, values=["Male", "Female"])
        self.DNA_combobox_gender.place(relx=0.5, rely=0.2, anchor=tk.CENTER)

        # create the profile frame
        splash_screen.set_text("Creating profile page")
        self.profile_frame = ctk.CTkFrame(master, corner_radius=0, fg_color="transparent")
        ProfileClass(self.profile_frame)

        # create the skills frame
        splash_screen.set_text("Creating skills page")
        self.skills_frame = ctk.CTkFrame(master, corner_radius=0, fg_color="transparent")
        SkillGUI(self.skills_frame)

        self.weather_frame = ctk.CTkFrame(master, corner_radius=0, fg_color="transparent")
        WeatherGUI(self.weather_frame)

        self.music_frame = ctk.CTkFrame(master, corner_radius=0, fg_color="transparent")
        Sounder(self.music_frame)

        # create the more frame
        self.more_frame = ctk.CTkFrame(master, corner_radius=0, fg_color="transparent")
        self.more_frame.grid(row=0, column=0, sticky=tk.NSEW)
        self.more_frame.grid_rowconfigure(8, weight=1)  # 7

        # Define navigation buttons and their properties
        self.more_buttons = [
            {"attribute": "more_weather_button", "text": "Weather", "image": self.image_weather_icon_image, "command": self.frame_more_weather_button_event, "tooltip": "View weather"},
            {"attribute": "more_news_button", "text": "News", "image": self.image_news_icon_image, "command": self.frame_2_button_event, "tooltip": "Read news"},
            {"attribute": "more_meditation_button", "text": "Meditation", "image": self.add_skills_image, "command": self.frame_3_button_event, "tooltip": "Start meditation"},
            {"attribute": "more_reminder_button", "text": "Reminder", "image": self.add_skills_image, "command": self.frame_DNA_button_event, "tooltip": "Set a reminder or todo"},
            {"attribute": "more_music_button", "text": "Music", "image": self.add_music_image, "command": self.frame_music_button_event, "tooltip": "Listen to music"},
            {"attribute": "more_workout_button", "text": "Workout", "image": self.add_skills_image, "command": self.frame_skills_button_event, "tooltip": "Start workout"},
            {"attribute": "more_productivity_button", "text": "Productivity", "image": self.add_skills_image, "command": self.frame_more_button_event, "tooltip": "Boost productivity"}
        ]

        # Create navigation buttons and assign attributes to self
        for index, nav_button in enumerate(self.more_buttons):
            button = ctk.CTkButton(
                self.more_frame,
                corner_radius=0,
                border_spacing=4,  # 10
                text=nav_button["text"],
                fg_color="transparent",
                text_color=("gray10", "gray90"),
                hover_color=("gray70", "gray30"),
                image=nav_button["image"],
                anchor="w",
                command=nav_button["command"],
                width=140,
                height=38,  # 28
            )
            setattr(self, nav_button["attribute"], button)

            # Configure column and row to expand and fill available space
            #self.more_frame.columnconfigure(index, weight=1, minsize=1) #minsize=120
            #self.more_frame.rowconfigure(0, weight=1)

            # Calculate row and column for the button
            row = index // 3  # Number of buttons per row
            column = index % 3  # Number of buttons in the current row

            button.grid(row=row, column=column, sticky=tk.NSEW)
            # Add tooltip
            # balloon = Balloon(self.more_frame)
            # balloon.bind_widget(button, balloonmsg=nav_button["tooltip"])

        # select default frame
        self.select_frame_by_name("home")

        if DEBUG_CHATBOT is None or DEBUG_CHATBOT is True:
            # Set up voice
            splash_screen.set_text("Set up voice")
            self.engine = pyttsx3_init()
            self.voices = self.engine.getProperty('voices')
            self.engine.setProperty('voice', self.voices[1].id)  # Index 1 for female voice
            self.engine.setProperty('rate', 150)  # Adjust rate to 150 words per minute
            self.engine.setProperty('volume', 0.7)  # Adjust volume to 70% of maximum
            self.engine.setProperty('pitch', 110)  # Adjust pitch to 110% of default
            del self.voices

        # Delete useless stuff
        splash_screen.set_text("Deleting useless stuff")
        del image_path
        del self.large_test_image
        del self.home_image
        del self.chat_image
        del self.add_user_image

        self.recognize_thread = None
        self.stoped_lisening = False
        self.message_count = 0
        self.is_lisening_wakeword = False

        if DEBUG_CHATBOT is None or DEBUG_CHATBOT is True:
            # Initialize chatbot
            splash_screen.set_text("Training the chatbot")
            self.chatbot = engine.MainEngine()
            #self.chatbot = Chatbot(splash_screen)
            #self.chatbot.train_bot()  # Train the chatbot

        if DEBUG_CHATBOT is None or DEBUG_CHATBOT is True:
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
            self.MessagesController = MessagesController(
                chat_frame=self.chat_frame,
                UserField=self.UserField,
                AITaskStatusLbl=self.AITaskStatusLbl,
                current_chat_bubble=self.current_chat_bubble,
                logo_image=self.logo_image
            )

        # Bind the 'Return' key event to the send_message method
        self.UserField.bind('<Return>', lambda event: self.MessagesController.send_message(None))

        self.wake_word_detecter = WakeWord(gui_callback=self.wake_word_callback)
        self.wake_word_detecter.start()

    def select_frame_by_name(self, name):
        frames = {
            "home": (self.home_frame, self.home_button),
            "frame_2": (self.second_frame, self.frame_2_button),
            "frame_3": (self.third_frame, self.frame_3_button),
            "frame_DNA": (self.DNA_frame, self.frame_DNA_button),
            "frame_profile": (self.profile_frame, self.frame_profile_button),
            "frame_skills": (self.skills_frame, self.frame_skills_button),
            "frame_more": (self.more_frame, self.frame_more_button),
            "frame_more_weather": (self.weather_frame, self.more_weather_button),
            "frame_more_music": (self.music_frame, self.more_music_button)
        }

        # Set button color for selected button
        for frame_name, (frame, button) in frames.items():
            button.configure(fg_color=("gray75", "gray25") if name == frame_name else "transparent")

        # Show selected frame and hide others
        for frame_name, (frame, _) in frames.items():
            if name == frame_name:
                # If the current frame name matches the selected frame name, show the frame
                frame.grid(row=0, column=1, sticky="nsew")
            else:
                # If the current frame name does not match the selected frame name, hide the frame
                frame.grid_forget()

    def debug_click(self):
        DebugGUI(self.master)

    def style_dropdown_click(self, event):
        self.entry.insert(1, event)
        apply_style(self.master, event)

    def wake_word_callback_END(self, user_input):
        """
        Callback function triggered when the user input for wake word processing is completed.

        Parameters:
            user_input (str): The user input recognized after the wake word is detected.
        """
        self.is_lisening_wakeword = False
        del self.w_gui
        self.frame_2_button_event()
        self.MessagesController.send_message(text=user_input)
        self.wake_word_detecter.unpause()

    def wake_word_callback(self, text):
        """Callback function triggered when the wake word is detected."""
        if self.is_lisening_wakeword == False:
            self.is_lisening_wakeword = True
            self.wake_word_detecter.pause()
            self.w_gui = WakeWordGUI(self.master, self.wake_word_callback_END)

    def chat_bubble_enable_event(self):
        """
        Enables or disables the chat bubble feature based on the user's selection.
        When chat bubble is enabled, the chat messages will be displayed in a custom frame.
        When chat bubble is disabled, the chat messages will be displayed in a regular text box.

        This method retrieves the value of the chat bubble switch, destroys the existing chat frame,
        and creates a new frame based on the selected chat bubble mode.

        Note: This method assumes the existence of `self.chat_bubble_switch_var` and `self.current_chat_bubble`.
        """
        print(self.chat_bubble_switch_var.get())
        print(self.current_chat_bubble)
        # Get the current value of the chat bubble switch
        value = self.chat_bubble_switch_var.get()

        # Check if the chat bubble switch is set to "on"
        if value == "on":
            self.current_chat_bubble = True

            # Destroy the existing chat frame and create a custom chat frame
            self.chat_frame.destroy()
            self.chat_frame = CTkXYFrame(self.root1, width=380, height=551, fg_color=chatBgColor, scrollbar_button_color='white')
            self.chat_frame.pack(padx=10)  #,fill="both", expand=False)

            # Update the chat frame and chat bubble mode in MessagesController
            self.MessagesController.chat_frame = self.chat_frame
            self.MessagesController.current_chat_bubble = self.current_chat_bubble
        else:
            # Disable chat bubble mode
            self.current_chat_bubble = False

            # Destroy the existing chat frame and create a regular text box for chat messages
            self.chat_frame.destroy()
            self.chat_frame = ctk.CTkTextbox(self.root1, width=380, height=551, fg_color=chatBgColor)
            self.chat_frame.pack(padx=10)

            # Update the chat frame and chat bubble mode in MessagesController
            self.MessagesController.chat_frame = self.chat_frame
            self.MessagesController.current_chat_bubble = self.current_chat_bubble

    def record(self, clear_chat=True, icon_display=True):
        """
        Records audio from the microphone and converts it to text using Google Speech Recognition.

        Args:
            clear_chat (bool): Flag indicating whether to clear the chat display. Defaults to True.
            icon_display (bool): Flag indicating whether to display an icon. Defaults to True.

        Returns:
            str: The recognized speech as lowercase text, or 'None' if an error occurred during speech recognition.
        """
        import speech_recognition as sr

        # Display status message
        print('\nListening...')
        self.AITaskStatusLbl.configure(text="Listening...")

        recognizer = sr.Recognizer()
        recognizer.dynamic_energy_threshold = False
        recognizer.energy_threshold = 4000

        with sr.Microphone() as source:
            # Adjust for ambient noise
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source)
            said = ""

            try:
                self.AITaskStatusLbl.configure(text="Processing...")
                # Convert audio to text using Google Speech Recognition
                said = recognizer.recognize_google(audio)
                print(f"\nUser said: {said}")
            except sr.UnknownValueError:
                print("Speech recognition could not understand audio")
                return 'None'
            except sr.RequestError as e:
                print(f"Could not request results from Google Speech Recognition service: {e}")
                return 'None'

        return said.lower()

    def voiceMedium(self):
        """
        Continuously records voice input and sends it as a message using MessagesController.
        This method is designed to run in a loop until the chatMode variable is set to 0.
        """
        while True:
          if chatMode == 0:
              print("chatmode")
              break
          query = self.record()
          if query == 'None': continue
          self.MessagesController.send_message(query)

    def changeChatMode(self):
        global chatMode
        if chatMode == 1:
            # If the current chat mode is 1 (voice mode)
            # Mute the volume (commented out)
            # appControl.volumeControl('mute')

            # Hide the VoiceModeFrame and show the TextModeFrame
            self.VoiceModeFrame.pack_forget()
            self.TextModeFrame.pack(fill=BOTH)

            # Set focus to the UserField
            self.UserField.focus()

            # Set the chat mode to 0 (text mode)
            chatMode = 0
            if type(self.recognize_thread) == type(Thread):
                self.recognize_thread.join()
        else:
            # If the current chat mode is not 1 (text mode)
            # Set the volume to full (commented out)
            # appControl.volumeControl('full')

            # Hide the TextModeFrame and show the VoiceModeFrame
            self.TextModeFrame.pack_forget()
            self.VoiceModeFrame.pack(fill=BOTH)

            # Set focus to the root window (commented out)
            # self.root.focus()

            # Set the chat mode to 1 (voice mode)
            chatMode = 1
            try:
                # pass
                if type(self.recognize_thread) != type(Thread):
                    self.recognize_thread = Thread(target=self.voiceMedium, daemon=True).start()
            except:
                pass

    def home_button_event(self):
        """
        Event handler for the home button.

        This method is responsible for selecting the 'home' frame when the home button is clicked.
        """
        self.select_frame_by_name("home")

    def frame_2_button_event(self):
        """Event handler for the chat button."""
        self.select_frame_by_name("frame_2")

    def frame_3_button_event(self):
        """Event handler for the settings button."""
        self.select_frame_by_name("frame_3")

    def frame_DNA_button_event(self):
        """Event handler for the DNA button."""
        self.select_frame_by_name("frame_DNA")

    def frame_profile_button_event(self):
        """Event handler for the profile button."""
        self.select_frame_by_name("frame_profile")

    def frame_skills_button_event(self):
        """Event handler for the skills button."""
        self.select_frame_by_name("frame_skills")

    def frame_more_button_event(self):
        """Event handler for the more button."""
        self.select_frame_by_name("frame_more")

    def frame_more_weather_button_event(self):
        """Event handler for the weather button."""
        self.select_frame_by_name("frame_more_weather")

    def frame_music_button_event(self):
        """Event handler for the music button."""
        self.select_frame_by_name("frame_more_music")

    def change_appearance_mode_event(self, new_appearance_mode):
        """Change the GUI appearance mode"""
        ctk.set_appearance_mode(new_appearance_mode)