from core.skill.base_skill import BaseSkill
from random import sample


class ChatSkill(BaseSkill):

    def __init__(self):
        super().__init__()
        self.intent = "ava_skills.chat"
        print(self.intent)
        self.active = True
        self.samples = [
            "hello",
            "lets chat",
            "who are you",
            "lets chat",

            "its good that you had fun",

            "hows it going",
            "whats your favorite hobby",
            "what can you do for me",



            "hi",
            "thank you",
            "what is your name",
            "thanks a lot",
            "i admire your creativity",
            "i did not mean to upset you",

            "i am so happy",
            "i love you",
            "i am so proud of you",

            "i am so excited",
            "i am so excited about your project",
            "i am so excited to talk to you",
            "what is your favorite food",
            "ask me anything",
            "what do you do for fun",
            "what are you doing",
            "write a python script that adds two numbers"
        ]

    def actAndGetResponse(self, **kwargs) -> str:
        return "CHAT"
