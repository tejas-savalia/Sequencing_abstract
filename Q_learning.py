
# coding: utf-8

# In[6]:

import numpy
import Environment as ev
import Agent
import random


	# In[7]:

#q values is a list of tuples of the form (position, action, value). This list gives the entire grid world
q_values = list()
for i in range(10):
    for j in range(10):
        for k in ev.get_legal_actions([i, j]):
            q_values.append([[i, j], k.tolist(), 0])
#print "Q Values", q_values
alpha = 0.1
gamma = 0.9


# In[8]:

def next_pos(curr_pos, action):
    next_pos, reward = ev.model_free_action(numpy.array(curr_pos), numpy.array(action))
    return next_pos, reward


# In[9]:

def q_states_from_pos(q_values, pos):
    q_state = list()
    for i in q_values:
        if i[0] == pos:
            q_state.append(i)
    return q_state


# In[10]:

def find_max_q(q_state):
    a = list()
    random.shuffle(q_state)
    for i in q_state:
        a.append(i[2])
    max_q = q_state[a.index(max(a))]
    return max_q


# In[11]:

def action(pos, q_values):
    q_state = q_states_from_pos(q_values, pos)
    #find the action that gives max q value 80% times. Rest, explore
    if random.uniform(0, 1) <= 0.8:
        max_q = find_max_q(q_state)
    else:
        max_q = random.choice(q_state)
    #Take the next step in the environment. Function returns new position and instant reward the environment gave
    new_q_pos, instant_reward = next_pos(pos, max_q[1])    
    new_state = q_states_from_pos(q_values, new_q_pos.tolist())
    #TD Upate to previous state, based on instant reward and belief of the value of the next state
    max_q[2] = q_update(max_q, new_state, instant_reward)
    #Update the beleif by updating in q_values. 
    q_values[q_values.index(max_q)] = max_q
    #if instant_reward != 0:
    #    new_state = q
    if instant_reward != 0:
        new_state = q_states_from_pos(q_values, [0,0])
    return new_state, q_values


# In[12]:

def q_update(state, new_state, instant_reward):
    max_over_new_state = find_max_q(new_state)
    #update q. Q value of current state-action pair; Q(s, a) is equal to it's previous estimate + alpha times temporal difference error
    state[2] = state[2] + alpha*(instant_reward + max_over_new_state[2] - state[2])
    updated_value = state[2]
    return updated_value


# In[19]:

new_pos = [0, 0]

def q_learn(iterations, q_values, new_pos):
	for i in range(iterations):
        
		new_state, q_values = action(new_pos, q_values)
		new_pos = new_state[0][0]
	return new_state, new_pos, q_values

#a = q_learn(10000, q_values, new_pos)	
#print "q_val",a[2]

