
# coding: utf-8

# In[2]:

import numpy


# In[3]:

grid_size = [10, 10]
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

print grid_world[:, 0]


# In[4]:

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
    grid_world[-1, 1] = 10

    return grid_world


# In[5]:

grid_structure = grid_struct(grid_size)


# In[72]:

def grid_world(state, action):
    #state is of the form [grid_pos, value]
    #action of the form "Left", "Right", "Up", "Down"
    action = action_vec(action)
    action = numpy.array(action)
    new_pos = numpy.add(state[0], action)
    new_state = numpy.array([new_pos, state[1]])
    #if the new state is the last state (right corner), return reward 10, otherwise return reward 0  
    print numpy.add(numpy.array(grid_size), numpy.array([-1, -1]))
    if numpy.array_equal(new_state[0], numpy.add(numpy.array(grid_size), numpy.array([-1, -1]))):
        reward = 10
    else:
        reward = 0
    #if the new state is outside the grid, return the old state and 0 reward otherwise return the state and its reward (if any)
    print new_state[0]
    if new_state[0].tolist() in grid_structure[:, 0].tolist():
        return (new_state, reward)
    else:
        return (state, 0)


# In[74]:

(state, reward) = grid_world(numpy.array([[1, 8], 0]), 'Up')
print state
print reward


# In[75]:

def action_vec(action):
    if action == 'Left':
        return numpy.array([-1, 0])
    elif action == 'Right':
        return numpy.array([1, 0])
    elif action == 'Up':
        return numpy.array([0, 1])
    elif action == 'Down':
        return numpy.array([0, -1])
    else:
        return numpy.array([0, 0])


# In[57]:

def state_verify(poss_state):
    if poss_state == grid_size:
        return poss_state
    if poss_state[0] < grid_size[0] and poss_state[1] < grid_size[1] and poss_state[0] >= 0 and poss_state [1] >= 0:
        allowed_state = poss_state
        return allowed_state
    return "State not valid"


# In[ ]:




# In[ ]:




# In[ ]:



