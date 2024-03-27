class value_function():
    vf={}

    def __init__(self,v):
        self.vf=v.copy()

    def get_value(self,t):
        if t in self.vf:
            return self.vf[t]
        else:
            return -float("inf")

class policy():
    
    vf={}
    mdp=None

    def __init__(self,v,mdp):
        self.vf=v.copy()
        self.mdp=mdp

    def select_action(self,s):
        #selectionne la meilleure action à faire pour l'état s, étant donné la fonction de valeur en attribut de la classe
        best_a=self.mdp.UP
        for a in self.mdp.get_actions(s):
            if self.get_reward(s, best_a) + self.get_second_term(s, best_a) < self.get_reward(s,a) + self.get_second_term(s, a):
                best_a = a
        return best_a 
    
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
            output += self.vf[next_state] * proba
        return output*self.mdp.get_discount_factor()