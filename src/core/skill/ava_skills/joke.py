from core.skill.base_skill import BaseSkill
import requests

class JokeSkill(BaseSkill):

    def __init__(self):
        super().__init__()
        self.intent = "ava_skills.joke"
        self.active = True
        self.samples = [
            "joke",
            "tell me a joke",
            "i want to laugh",
            "tell me something funny",
            "i want to hear a joke",
            "i want a joke",
            "joke please",
            "tell me a joke please",
            "i want to hear something funny",
            "i want to hear something funny please",
        ]

    def actAndGetResponse(self, **kwargs) -> str:
        r = requests.get("https://v2.jokeapi.dev/joke/Any")
        j = r.json()
        if j["type"] == "single":
            return j["joke"]
        else:
            return j["setup"] + " " + j["delivery"]
