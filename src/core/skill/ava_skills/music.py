from core.skill.base_skill import BaseSkill
from random import sample


class MusicSkill(BaseSkill):

    def __init__(self):
        super().__init__()
        self.intent = "ava_skills.music"
        print(self.intent)
        self.active = True
        self.samples = [
            "could you please play some music",
            "play music",
            "music",
            "music please",
            "i want to listen to music",
            "play some music",
            "i want to hear some music",
            "i want to hear some music please",
        ]

    def actAndGetResponse(self, **kwargs) -> str:
        return "MUSIC"
