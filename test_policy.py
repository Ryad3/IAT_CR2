from gridworld import *

mdp = GridWorld ()

print("states:",mdp.get_states())
print("terminal states:",mdp.get_goal_states()) 
print("actions:",mdp.get_actions())
print(mdp.get_transitions(mdp.get_initial_state(),mdp.UP))

def policy_custom(state):
    x,y = state
    if y < mdp.height - 1:
        return mdp.UP
    else:
        return mdp.RIGHT

while(1):
    state=mdp.get_initial_state()
    new_state,_ = mdp.execute(state,policy_custom(state))
    mdp.initial_state=new_state 
    mdp.visualise()