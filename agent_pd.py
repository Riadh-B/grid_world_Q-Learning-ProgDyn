import numpy as np
from environment.grid_env import Environment
import copy


BOARD_COLS = 4
BOARD_ROWS = 3
EPSILON = 0.01
GAMMA = 0.9

UP = 'up'
DOWN = 'down'
LEFT = 'left'
RIGHT = 'right'
ACTIONS = { \
    UP:[UP, LEFT, RIGHT], \
    DOWN:[DOWN, LEFT, RIGHT], \
    LEFT:[LEFT, UP, DOWN], \
    RIGHT:[RIGHT, UP, DOWN]}
nA=4
nS=BOARD_COLS *BOARD_ROWS 
tot_policy=[]
for s in range(nS):
    policy ={str(act):1/len(ACTIONS) for act in ACTIONS}
    tot_policy.append(policy)
grid2state_dict={(s//BOARD_COLS, s%BOARD_COLS):s for s in range(nS)}

# Init V(s)
V = np.zeros((BOARD_ROWS, BOARD_COLS), dtype='float16')

# init actions probabilities 
# [prob_main_action, prob_first_alternative_action, prob_second_alternative_action]
p = [0.8, 0.1, 0.1]

#****************************Policy Iteration***********************

def policy_evaluation(tot_policy, GAMMA, EPSILON):
    delta  = 1
    while(delta > EPSILON):
        delta = 0
        for row in range(BOARD_ROWS):
            for col in range(BOARD_COLS):
                state_value_old = V[row][col]
                for action in ACTIONS:
                    proba_act=tot_policy[grid2state_dict[(row,col)]][action]
                    new_state_value = 0.0
                    for choix, prob in zip(ACTIONS[action], p):
                        env = Environment(state = (row, col), deterministic=True)
                        next_state = env.nextPosition(choix)
                        #print("state = ({}, {})", row, col)
                        #print("action = " + choix)
                        #print("next-state = " + str(next_state))
                        #print()
                        reward = env.giveReward()
                        new_state_value += proba_act*prob * (reward + GAMMA * V[next_state[0]][next_state[1]])
                V[row][col] = new_state_value 
                v_change = np.abs(state_value_old -  new_state_value)
                delta = np.maximum(delta, v_change)
        print("\nDelta = " + str(delta) + "\n")
    print("V = " + str(V))
    return(V)

#Obtain Q_Pi from V_Pi
def q_from_v (V, state, GAMMA):
    q = {str(act):0 for act in ACTIONS}
    for action in q:
        for choix, prob in zip(ACTIONS[action], p):
            env = Environment(state = state, deterministic=True)
            next_state = env.nextPosition(choix)
            reward = env.giveReward()
            q[action] += prob * (reward + GAMMA * V[next_state[0]][next_state[1]])
    return q

def get_state_coord(nCol, state):
    return (state//nCol, state-(nCol * (state//nCol)))
"""V = policy_evaluation(tot_policy, GAMMA, EPSILON)
Q = []
for s in range(nS): 
    print("s = " + str(s))
    print('coods of {} = {}')
    print(s)
    print(get_state_coord(BOARD_COLS,s))
    result = q_from_v(V, get_state_coord(BOARD_COLS,s),GAMMA)
    Q.append(result)
print("Action-Value Function:")
print(Q)"""

def cle_max(dico,maxinit = -1000000):
  max = maxinit
  for x in dico.keys():
    if dico[x]>max :
       clemax,max = x,dico[x]
  return clemax

def policy_improvement(V, GAMMA):
    policy ={str(act):0 for act in ACTIONS}
    tot_policy=[]
    Q = []
    for s in range(nS): 
        result = q_from_v(V, get_state_coord(BOARD_COLS,s),GAMMA)
        Q.append(result)
    print("Action-Value Function:")
    print(Q)w
    for s in range(nS):
        q =q_from_v(V, get_state_coord(BOARD_COLS,s),GAMMA)
        print(q)
        # OPTION 1: construct a deterministic policy 
        key_max=cle_max(q,maxinit = -1000000)
        policy[key_max]=1
        tot_policy.append(policy)
        print(tot_policy)
        # OPTION 2: construct a stochastic policy that puts equal probability on maximizing actions
        #best_a = np.argwhere(q==np.max(q)).flatten()
        #policy[row][col] = np.sum([np.eye(nA)[i] for i in best_a], axis=0)/len(best_a) 
    
    return tot_policy



def policy_iteration(GAMMA, EPSILON):
    tot_policy=[]
    for s in range(nS):
        policy ={str(act):1/len(ACTIONS) for act in ACTIONS}
        tot_policy.append(policy)
    V = policy_evaluation (tot_policy, GAMMA, EPSILON)
    while True:
        V = policy_evaluation (tot_policy, GAMMA, EPSILON)
        new_policy = policy_improvement(V, GAMMA)   
        print("V en cours",V)
        # OPTION 1: stop if the policy is unchanged after an improvement step
        if (new_policy == tot_policy):
            break
        
        # OPTION 2: stop if the value function estimates for successive policies has converged
        # if np.max(abs(policy_evaluation(Environment, policy) - policy_evaluation(env, new_policy))) < EPSILON:
        #    break
        
        tot_policy = copy.copy(new_policy)
    return tot_policy, V

# obtain the optimal policy and optimal state-value function
policy_pi, V_pi = policy_iteration(GAMMA, EPSILON)
print("***************************Policy Iteration*******************")
print("\nValue State :")
print(V_pi ,"\n")
# print the optimal policy
print("\nOptimal Policy :")
print(policy_pi,"\n")

#plot_values(V_pi)"""

#******************************************** Value Iteration**************************************************************************
"""
delta  = 1 # a value greater than EPSILON to kick off the process
while(delta > EPSILON):
    delta = 0
    for row in range(BOARD_ROWS):
        for col in range(BOARD_COLS):
            state_value_old = V[row][col]
            values = [] # stores the value of each action (for a single state)
            for action in ACTIONS:
                new_state_value = 0.0
                for prob_action, prob in zip(ACTIONS[action], p):
                    env = Environment(state = (row, col), deterministic=True)
                    next_state = env.nextPosition(prob_action)
                    #print("state = ({}, {})", row, col)
                    #print("action = " + prob_action)
                    #print("next-state = " + str(next_state))
                    #print()
                    reward = env.giveReward()
                    new_state_value += prob * (reward + GAMMA * V[next_state[0]][next_state[1]])
                values.append(new_state_value)
            V[row][col] = np.max(values)
            v_change = np.abs(state_value_old - np.max(values))
            delta = np.maximum(delta, v_change)
    print("\nDelta = " + str(delta) + "\n")
print("***************Value Iteration**********************")
print("V = " + str(V))"""