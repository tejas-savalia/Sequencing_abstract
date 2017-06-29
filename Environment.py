
# coding: utf-8

# In[2]:

import numpy
import random


# In[3]:

grid_size = [10, 10]


# In[4]:

def state_verify(poss_state):
    if poss_state == grid_size:
        return poss_state
    if poss_state[0] < grid_size[0] and poss_state[1] < grid_size[1] and poss_state[0] >= 0 and poss_state [1] >= 0:
        allowed_state = poss_state
        return allowed_state
    return "State not valid"


# In[ ]:





# In[6]:

def grid_struct(grid_size):
    position = list()
    grid = list()
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            position.append([i, j])
            grid.append(0)
    grid_arr = list()
    for i in range(len(grid)):
        grid_arr.append([position[i], grid[i]])
    grid_world = numpy.array(grid_arr)
    grid_world[55][1] = -10
    grid_world[99][1] = 10
    return grid_world


# In[7]:

grid_structure = grid_struct(grid_size)

# In[8]:

def get_legal_actions(curr_pos):
    poss_actions = numpy.array([[-1, 0], [1, 0], [0, 1], [0, -1]])
    legal_actions = list()
    for i in poss_actions:
        if numpy.add(curr_pos, i).tolist() in grid_structure[:, 0].tolist():
            legal_actions.append(i)
    return legal_actions


# In[9]:

def grid_world(state, action):
    #state is of the form [grid_pos, value]
    #action of the form "Left", "Right", "Up", "Down"
    action = numpy.array(action)
    action = transition_prob(state[0], action)
    new_pos = numpy.add(state[0], action)
    new_state = numpy.array([new_pos, state[1]])
    #if the new state is the last state (right corner), return reward 10, otherwise return reward 0  
    reward = grid_structure[(new_state[0][0]*10 + new_state[0][1])][1]
    return new_state, reward

def transition_prob(pos, action):
    random_no = random.uniform(0, 1)
    if random_no <= 0.8:
        return action
    else:
        return random.choice(get_legal_actions(pos))
# In[10]:

(state, reward) = grid_world(numpy.array([[3, 4], 0]), [0, 1])


# In[11]:




# In[12]:




# In[13]:

def model_free_action(pos, action):
    new_pos = numpy.add(pos, action)
    reward = grid_structure[(new_pos[0]*10 + new_pos[1])][1]
    return new_pos, reward


# In[15]:




# In[ ]:



