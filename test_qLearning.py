from gridworld import *
from q_learning import *
from fnc_value import *

mdp = GridWorld()

print("states:",mdp.get_states())
print("terminal states:",mdp.get_goal_states()) 
print("actions:",mdp.get_actions())
print(mdp.get_transitions(mdp.get_initial_state(),mdp.UP))

qL = q_agent(mdp)
qL.solve()
print(qL.Q)

q_fnc = q_function(qL.Q)

mdp.visualise_q_function(q_fnc)