#from q_function import *
import random 
from matplotlib import pyplot as plt
from collections import defaultdict

class q_agent:

    def __init__(self, mdp):# and here...
        self.mdp = mdp
        self.Q = defaultdict(lambda: 0)
        self.epsilon = 0.1
        self.alpha = 0.2
        self.nbr_episodes = 5000

    def greedy(self,s):
        #greedy functions returning best action to pick in a state
        action_max = max([(self.Q[(s, a)], a) for a in self.mdp.get_actions(s)], key=lambda x: x[0])[1]
        return action_max

    def epsilon_greedy(self, state):
        if random.random() < 1 - self.epsilon:
            return random.choice(self.mdp.get_actions(state))
        else:
            return self.greedy(state)

    def solve(self):
        #main solving loop
        for i in range(self.nbr_episodes):
            state = self.mdp.get_initial_state()
            while not self.mdp.is_terminal(state):
                a = self.epsilon_greedy(state)
                next_s, reward = self.mdp.execute(state, a)
                delta = self.get_delta(reward, self.Q[(state, a)], next_s)
                self.Q[(state, a)] = self.Q[(state, a)] + (self.alpha * delta)
                state = next_s

            

    def get_delta(self, reward, q_value, next_state):
        #Calculate the delta for the update
        return reward + (self.mdp.get_discount_factor() * self.state_value(next_state)) - q_value


    def state_value(self, state):
        #Get the value of a state
        maxi = max([self.Q[(state, a)] for a in self.mdp.get_actions(state)])
        return maxi
