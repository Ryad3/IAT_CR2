from mdp import *


class dp_agent():
    mdp=None
    v=None
    v_bis=None
    
    def __init__(self,mdp): #and here...
        self.mdp=mdp
        self.v = {el : 0.0 for el in self.mdp.get_states()}
        self.v_bis = {el : 0.0 for el in self.mdp.get_states()}
        self.epsilon = 0.01

    def get_value(self,s):
        #return the value of a specific state s according to value function v
        return self.v[s]
        
    def get_width(self):
        #return the absolute norm between two value functions
        return max([abs(self.v[s]-self.v_bis[s]) for s in self.v])

    def solve(self):
        #main solving loop
        while True:
            self.v_bis = self.v.copy()
            for s in self.mdp.get_states():
                self.update(s)
            if self.get_width() < self.epsilon:
                break

    def update(self,s):
        #updates the value of a specific state s
        self.v[s] = max([self.get_reward(s,a) + self.get_second_term(s, a) for a in self.mdp.get_actions(s)])

    def get_reward(self, s, a):
        #returns the next state when taking action a from state s
        output = 0
        for next_state, proba in self.mdp.get_transitions(s, a):
            output += self.mdp.get_reward(s, a, next_state) * proba
        return output

    def get_second_term(self, s, a):
        #returns the second term of the Bellman equation
        output = 0
        for next_state, proba in self.mdp.get_transitions(s, a):
            output += self.v_bis[next_state] * proba
        return output*self.mdp.get_discount_factor()