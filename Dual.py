
# coding: utf-8

# In[1]:

import numpy
import Agent as mb
import Environment as ev
import Q_learning as mf


# Take action in model based "Agent". Get a state value pair for a particular action. Use that state action to update Q values.
# At a later point, do q learning

# In[2]:

def get_state_evaluation(internal_model, current_state):
    curr_pos = current_state[0]
    state_action_value = list()
    for i in ev.get_legal_actions(curr_pos):
        next_pos = numpy.add(numpy.array(curr_pos), numpy.array(i))
        next_state = internal_model[(10*next_pos[0] + next_pos[1])]
        next_pos_value = mb.depth_limited_search(3, 0, internal_model, next_state)
        state_action_value.append([next_pos.tolist(), i.tolist(), next_pos_value])
    return state_action_value


# In[3]:

def get_q_values(q_values, internal_model):
    for i in q_values:
        next_pos = numpy.add(i[0], i[1])
        next_state = internal_model[10*next_pos[0] + next_pos[1]]
        next_state_value = mb.depth_limited_search(3, 0, internal_model, next_state)
        i[2] = next_state_value
    return q_values

def get_internal_model(q_values, internal_model):
    for i in q_values:
        next_pos = numpy.add(i[0], i[1])
        internal_model[10*next_pos[0] + next_pos[1]][1] = 0.5*internal_model[10*next_pos[0] + next_pos[1]][1] + 0.5*i[2]
    return internal_model
# In[4]:

def dual(q_values, internal_model, mf_iter, mb_iter):
    #Use state space search to learn the internam model. Use that internal model
    #to get q values. Use those q values to do q learning. Use updated q values to form
    #internal model
    a = mb.mb_learn(mb_iter, mb.internal_model[0], [mb.internal_model[0][0]], mb.internal_model)
    internal_model = a[3]
    q_values = get_q_values(mf.q_values, internal_model)
    q_values = mf.q_learn(mf_iter, q_values, [0, 0])[2]
    internal_model = get_internal_model(q_values, internal_model)
    return q_values, internal_model

# In[ ]:
#A = dual(mf.q_values, mb.internal_model, 10000, 10)
q_learn = mf.q_learn(1000000, mf.q_values, [0, 0])[2]

# In[ ]:




# In[ ]:




# In[ ]:



