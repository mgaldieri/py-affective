__author__ = 'mgaldieri'
import numpy as np


class OCEAN():
    def __init__(self):
        self.openess = 0.0
        self.conscientiousness = 0.0
        self.extraversion = 0.0
        self.agreeableness = 0.0
        self.neuroticism = 0.0

    def pad(self, mode='native'):
        if mode == 'native':
            pass
        elif mode == 'vector':
            pass
        else:
            raise Exception('Invalid mode')


class OCC():
    def __init__(self, ):
        self.admiration = 0.0
        self.joy = 0.0
        self.relief = 0.0
        self.love = 0.0
        self.hope = 0.0
        self.happiness = 0.0
        self.gratification = 0.0
        self.gratitude = 0.0
        self.pride = 0.0
        self.gloat = 0.0
        self.satisfaction = 0.0
        self.sympathy = 0.0

        self.blame = 0.0
        self.disappointment = 0.0
        self.fear = 0.0
        self.confirmed_fear = 0.0
        self.hate = 0.0
        self.pity = 0.0
        self.rage = 0.0
        self.remorse = 0.0
        self.disavowal = 0.0
        self.resentment = 0.0
        self.shame = 0.0


class PAD():
    def __init__(self):
        self.pleasure = 0.0
        self.arousal = 0.0
        self.dominance = 0.0

        self.states = {
            'exuberant': np.array([1.0, 1.0, 1.0]),
            'dependant': np.array([1.0, 1.0, -1.0]),
            'relaxed': np.array([1.0, -1.0, 1.0]),
            'amenable': np.array([1.0, -1.0, -1.0]),

            'bored': np.array([-1.0, -1.0, -1.0]),
            'disdainful': np.array([-1.0, -1.0, 1.0]),
            'anxious': np.array([-1.0, 1.0, -1.0]),
            'hostile': np.array([-1.0, 1.0, 1.0])
        }

    def state(self):
        mood = np.array([self.pleasure, self.arousal, self.dominance])
        states = self.states.keys()
        values = self.states.values()


class Agent():
    def __init__(self, personality):
        self.personality = personality if personality else OCEAN()
        self.mood = np.array(np.zeros(3))

    def _occ2pad(self, occ):
        pass

    def _pad2scalar(self, pad):
        pass

    def put(self, val):
        pass

    def get(self, mode='pad'):
        pass