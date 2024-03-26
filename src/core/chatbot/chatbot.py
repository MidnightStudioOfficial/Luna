"""
Chatbot System

This script implements a chatbot system that combines a Conversational Engine and a ChatterBot-based chatbot.
The Conversational Engine handles natural language interactions, while the ChatterBot is used to provide responses 
to user inputs when the Conversational Engine does not find an appropriate response.

It is recommended to review the required data files, ensure proper file paths, and have the necessary data available before running the script.
"""

Debug = False


print('Importing engine2 (This may take a while!)')
from core.engine.EngineCore import conversation as Engine2
from os.path import isfile
import logging
import json
print("Importing DONE")


if Debug != True:
    trainingdata = 'Data/training.csv'
    articulationdata = 'Data/articulations.csv'
else:
    trainingdata = 'Data/training_dev.csv'
    articulationdata = 'Data/articulations_dev.csv'

logging.basicConfig(level=logging.INFO)


class ChatbotProfile:
      def __init__(self) -> None:
        print("Importing brain (This may take a while!)")
        from core.brain.brain import Brain
        self.profile_data = {
              "name": None,
              "gender": None,
              "brain": {
                   "traits": [],
                   "mood": None,
                   "thought": None,
                   "memory": {}
              }
        }
        self.brain = Brain()
        self.brain.start()

      def update_profile(self):
          self.profile_data["brain"]["mood"] = self.brain.mood
          self.profile_data["brain"]["thought"] = self.brain.thought

      def _set_profile_data(self):
          self.brain.mood = self.profile_data["brain"]["mood"]
          self.brain.thought = self.profile_data["brain"]["thought"]

      def load_profile(self):
        # Open the JSON file
        try:
         with open('./Data/chatbot/profile.json', 'r') as f:
             # Load the contents of the file as a Python object
             data = json.load(f)
             self.profile_data["name"] = data["name"]
             self.profile_data["gender"] = data["gender"]
             self.profile_data["brain"]["traits"] = data["brain"]["traits"]
             self.profile_data["brain"]["mood"] = data["brain"]["mood"]
             self.profile_data["brain"]["thought"] = data["brain"]["thought"]
             self.profile_data["brain"]["memory"] = data["brain"]["memory"]
        except FileNotFoundError:
           print("File not found")
        except json.JSONDecodeError:
           print("Invalid JSON syntax")
        self._set_profile_data()

      def save_profile(self):
       self.update_profile()
       try:
            with open('./Data/chatbot/profile.json', 'w') as f:
                # Load the contents of the file as a Python object
                data = json.dump(self.profile_data, f)
       except FileNotFoundError:
           print("File not found")
       except json.JSONDecodeError:
           print("Invalid JSON syntax")


class Chatbot:
    def __init__(self, splash_screen) -> None:
        # Initialize the conversational engine and conversation
        splash_screen.set_text("Initializing the conversational engine and conversation")
        
        self.engine = Engine2()
    def train_bot(self) -> None:
        """
        Train the chatbot if it doesn't already exist.
        """
        logging.debug("Training bot")
    def get_skill(self, input_text) -> bool:
        """
        Check if the input is a skill.

        Parameters:
            input_text (str): The user input text.

        Returns:
            bool: True if the input is a skill, False otherwise.
        """
        if input_text != "CHAT":
            return True
        else:
            return False

    def get_response(self, input_text):
        """
        Get a response from the conversational engine or chatbot.

        Parameters:
            input_text (str): The user input text.

        Returns:
            str: The response from the conversational engine or chatbot.
        """
        # Get a response from the conversational engine or chatbot
        payload = self.engine.getIntent(input_text) # Get the intent from Engine2 based on the input
        response = payload.get('intent') # Extract the intent from the payload

        is_skill = self.get_skill(response)  # Check if the response is a skill using the get_skill method
        if is_skill: # If it's a skill
            return response # Return the response as it is a skill

        # If not a skill, get a response from the chatbot
        bot = self.chatBot.get_response(text=input_text, search_text=input_text)
        print(bot.text)
        return bot.text
