from gridworld import *
from dynamic_programming import *
from fnc_value import *

mdp = GridWorld(noise=0)

print("states:",mdp.get_states())
print("terminal states:",mdp.get_goal_states()) 
print("actions:",mdp.get_actions())
print(mdp.get_transitions(mdp.get_initial_state(),mdp.UP))

dp_a = dp_agent(mdp)
dp_a.solve()
print(dp_a.v)

fnc = value_function(dp_a.v)
plc = policy(dp_a.v, mdp)

mdp.visualise_value_function(fnc)
mdp.visualise_policy(plc)