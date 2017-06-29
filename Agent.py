
# coding: utf-

# In[1]:

import numpy
import random
import Environment as ev


# In[2]:

internal_model = list()
grid_size = 10
for i in range(grid_size):
    for j in range(grid_size):
        internal_model.append([[i, j], 0])
internal_model = numpy.array(internal_model)
internal_model[10, 1] = 5
internal_model[15, 1] = 10


# In[3]:

def model_update(internal_model, instant_reward, state):
    internal_model[state[0][0]*10 + state[0][1]][1] = instant_reward
    return internal_model


# In[4]:

gamma = 0.9


# In[5]:

def depth_limited_search(max_depth, curr_depth, internal_model, curr_state):
    value = 0
    curr_pos = curr_state[0]
    curr_val = curr_state[1]
    legal_actions = ev.get_legal_actions(curr_pos)
    curr_depth += 1
    if max_depth <= curr_depth:
        return curr_val
    else:
        for i in legal_actions:
            new_pos = numpy.add(curr_pos, i)
            new_state = internal_model[(10*new_pos[0] + new_pos[1])]
            value += curr_val + gamma * depth_limited_search(max_depth, curr_depth, internal_model, new_state)
        return value


# In[ ]:




# In[6]:

def model_based_action_selection(curr_state, visited_pos, internal_model):
    curr_val = curr_state[1]
    curr_pos = curr_state[0]
    legal_actions = ev.get_legal_actions(curr_pos)
    next_state_values = list()
    actions_to_next_states = list()
    for i in legal_actions:
        new_pos = numpy.add(curr_pos, i)
        if any(numpy.equal(numpy.array(visited_pos),new_pos).all(1)):
            continue
        new_state = internal_model[(10*new_pos[0] + new_pos[1])]
        next_state_values.append(depth_limited_search(5, 0, internal_model, new_state))
        actions_to_next_states.append(i)
    if next_state_values == []:
        action = random.choice(legal_actions)
    else:
        value_action = list(zip(next_state_values, actions_to_next_states))
        random.shuffle(value_action)
        next_state_values[:], actions_to_next_states[:] = zip(*value_action)
        if random.uniform(0, 1) < 0.8:
            action = actions_to_next_states[next_state_values.index(max(next_state_values))]        
        else:
            action = random.choice(actions_to_next_states)
    return action


# In[7]:

def take_action(state, action):
    (new_state, reward) = ev.grid_world(state, action)
    return new_state, reward


# In[8]:

curr_state = internal_model[0]
visited_pos = [curr_state[0]]
def mb_learn(iterations, curr_state, visited_pos, internal_model):
	reward = 0
	for i in range(iterations):
		action = model_based_action_selection(curr_state, visited_pos, internal_model)
		state, instant_reward = take_action(curr_state, action)
		internal_model = model_update(internal_model, instant_reward, state)
		visited_pos = numpy.vstack((visited_pos, state[0]))
		curr_state = state
		reward += instant_reward
	return state, curr_state, action, internal_model, reward, visited_pos
