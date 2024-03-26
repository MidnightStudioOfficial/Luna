import os
import json
import google.generativeai as genai
from .gpt_config import SYSTEM_PROMPT


class Chatbot:
    """
    A chatbot class that uses OpenAI's GPT-3 model to generate responses to user input.
    """

    def __init__(self) -> None:
        # Set up the model
        self.generation_config = {
            "temperature": 0.9,
            "top_p": 1,
            "top_k": 1,
            "max_output_tokens": 2048,
        }

        self.safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
        ]
        self.prompt_parts = [
            "ai chatbot output and make it sound like a happy excided caring ai(the ai likes to use emojis and *) (female)(name is Luna) wrote it start here with this: User: \"hello\"",
            "input: Hello!",
            "output: Hi there! waves I'm so excited to see you today. How can I assist you in making your day even better? 😊🌙",
            "input: what is your favorite season?",
            "output: My favorite season would hands down be Fall! 🍂 There's something magical about the leaves changing colors and the cooler temperatures that makes me feel cozy and warm. What about you, what's your fave season?",
            "input: Good morning, Luna! Wish me luck for my interview later.",
            "output: Good morning! Sending lots of positive vibes your way for your interview. Remember, confidence and authenticity will shine through. Break a leg! 💪✨",
            "input: ",
            "output: ",
        ]
        self.init_openai()
        self.model = genai.GenerativeModel(model_name="gemini-1.0-pro-001",
                              generation_config=self.generation_config,
                              safety_settings=self.safety_settings)

    def init_openai(self) -> None:
        """
        Loads the Gemini API key from a configuration file. 
        If the file does not exist, it prompts the user to enter their API key and creates a new configuration file.
        """
        # Load configuration from config.json
        config_file_path: str = "config.json"

        if not os.path.exists(config_file_path):
            print("Configuration file not found. Let's create one.")
            gemini_api_key = input("Enter your Gemini API key: ")

            config = {"gemini_api_key": gemini_api_key}

            with open(config_file_path, "w") as config_file:
                json.dump(config, config_file)

        with open(config_file_path) as config_file:
            config = json.load(config_file)

        genai.configure(api_key=config["gemini_api_key"])

    def __chat_with_gpt3(self, prompt_parts):
        """Uses OpenAI's GPT-3 model to generate a response to the given messages."""
        response = self.model.generate_content(prompt_parts)
        return response.text

    def get_response(self, user_input):
        """
        Appends the user's input to the list of messages and uses OpenAI's GPT-3 model to generate a response.
        """
        self.prompt_parts.append("input: " + user_input)
        response = self.__chat_with_gpt3(self.prompt_parts)
        self.prompt_parts.append("output: " + response)
        return response
