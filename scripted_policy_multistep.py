import numpy as np
import random
import pprint
import matplotlib.pyplot as plt
from moviepy.editor import ImageSequenceClip
import threading
import os
import time
import pickle
from PIL import Image
import pybullet
import pybullet_data

class ScriptedPolicy():

  def __init__(self, env, PICK_TARGETS, PLACE_TARGETS):
    self.env = env
    self.PICK_TARGETS = PICK_TARGETS
    self.PLACE_TARGETS = PLACE_TARGETS

  def prompt_augs(self, prompt):
    prompt_aug = [prompt]
    if 'Pick' in prompt and not 'Pick up' in prompt:
        prompt_aug.append(prompt.replace('Pick', 'Pick up'))
        prompt_aug.append(prompt.replace('Pick', 'Grab'))
        prompt_aug.append(prompt.replace('Pick', 'Get'))
    if 'Pick up' in prompt:
        prompt_aug.append(prompt.replace('Pick up', 'Pick'))
        prompt_aug.append(prompt.replace('Pick up', 'Grab'))
        prompt_aug.append(prompt.replace('Pick up', 'Get'))
    if 'Put' in prompt:
        prompt_aug.append(prompt.replace('Put', 'Place'))
    if 'place' in prompt:
        prompt_aug.append(prompt.replace('place', 'put'))
        prompt_aug.append(prompt.replace('place', 'drop'))
        prompt_aug.append(prompt.replace('place', 'set'))
    if 'blocks' in prompt:
        prompt_aug.append(prompt.replace('blocks', 'cubes'))
        prompt_aug.append(prompt.replace('blocks', 'boxes'))
    if 'bowls' in prompt:
        prompt_aug.append(prompt.replace('bowls', 'cups'))
    if 'bowl' in prompt:
        prompt_aug.append(prompt.replace('bowl', 'cup'))
    if 'a ' in prompt:
        prompt_aug.append(prompt.replace('a ', 'any '))
    return prompt_aug
  
  def generate_stack_demo(self, item_name):
    blocks = [k for k in self.PICK_TARGETS.keys() if item_name in k]
    prompt = 'Stack the {}s'.format(item_name)

    items1 = blocks.copy()
    action1 = ('pick', np.random.choice(blocks, 1).tolist())
    blocks.remove(action1[1][0])

    items2 = blocks.copy()
    action2 = ('place', np.random.choice(blocks, 1).tolist())
    blocks.remove(action2[1][0])

    items3 = blocks.copy()
    action3 = ('pick', np.random.choice(blocks, 1).tolist())
    blocks.remove(action3[1][0])

    items4 = action1[1]
    action4 = ('place', action1[1])

    items5 = blocks.copy()
    action5 = ('pick', np.random.choice(blocks, 1).tolist())
    blocks.remove(action5[1][0])

    items6 = action3[1]
    action6 = ('place', action3[1])

    items = [items1, items2, items3, items4, items5, items6]
    actions = [action1, action2, action3, action4, action5, action6]

    #print(prompt)
    #print(items)
    #print(actions)
    return prompt, items, actions

  def generate_block_same_corner_demo(self):
    blocks = [k for k in self.PICK_TARGETS.keys() if 'block' in k]
    corners = [k for k in self.PLACE_TARGETS.keys() if 'corner' in k]
    prompt = 'Put the blocks in the same corner'

    items1 = blocks.copy()
    action1 = ('pick', np.random.choice(blocks, 1).tolist())
    blocks.remove(action1[1][0])

    items2 = corners.copy()
    action2 = ('place', np.random.choice(corners, 1).tolist())
    corners.remove(action2[1][0])

    items3 = blocks.copy()
    action3 = ('pick', np.random.choice(blocks, 1).tolist())
    blocks.remove(action3[1][0])

    items4 = action1[1]
    action4 = ('place', items4)

    items5 = blocks.copy()
    action5 = ('pick', np.random.choice(blocks, 1).tolist())
    blocks.remove(action5[1][0])

    items6 = action3[1]
    action6 = ('place', action3[1])

    items7 = blocks.copy()
    action7 = ('pick', np.random.choice(blocks, 1).tolist())
    blocks.remove(action7[1][0])

    items8 = action5[1]
    action8 = ('place', action5[1])

    items = [items1, items2, items3, items4, items5, items6, items7, items8]
    actions = [action1, action2, action3, action4, action5, action6, action7, action8]

    #print(prompt)
    #print(items)
    #print(actions)

    return prompt, items, actions

  def generate_block_diff_corner_demo(self):
    blocks = [k for k in self.PICK_TARGETS.keys() if 'block' in k]
    corners = [k for k in self.PLACE_TARGETS.keys() if 'corner' in k]
    prompt = 'Put the blocks in different corners'
    
    items1 = blocks.copy()
    action1 = ('pick', np.random.choice(blocks, 1).tolist())
    blocks.remove(action1[1][0])

    items2 = corners.copy()
    action2 = ('place', np.random.choice(corners, 1).tolist())
    corners.remove(action2[1][0])

    items3 = blocks.copy()
    action3 = ('pick', np.random.choice(blocks, 1).tolist())
    blocks.remove(action3[1][0])

    items4 = corners.copy()
    action4 = ('place', np.random.choice(corners, 1).tolist())
    corners.remove(action4[1][0])

    items5 = blocks.copy()
    action5 = ('pick', np.random.choice(blocks, 1).tolist())
    blocks.remove(action5[1][0])

    items6 = corners.copy()
    action6 = ('place', np.random.choice(corners, 1).tolist())
    corners.remove(action6[1][0])

    items7 = blocks.copy()
    action7 = ('pick', np.random.choice(blocks, 1).tolist())
    blocks.remove(action7[1][0])

    items8 = corners.copy()
    action8 = ('place', np.random.choice(corners, 1).tolist())
    corners.remove(action8[1][0])

    items = [items1, items2, items3, items4, items5, items6, items7, items8]
    actions = [action1, action2, action3, action4, action5, action6, action7, action8]

    #print(prompt)
    #print(items)
    #print(actions)

    return prompt, items, actions

  def generate_pick_demo(self, item_name):
    blocks = [k for k in self.PICK_TARGETS.keys() if item_name in k]
    prompt = 'Pick up a {}'.format(item_name)
    items = [blocks]
    actions = [('place', np.random.choice(blocks, 1).tolist())]

    #print(prompt)
    #print(items)
    #print(actions)

    return prompt, items, actions

  def generate_pick_place_demo(self):
    all_pick_items = [k for k in self.PICK_TARGETS.keys() if 'block' in k]
    all_place_items = list(self.PLACE_TARGETS.keys())
    pick_item = random.choice(all_pick_items)
    all_place_items.remove(pick_item)
    place_item = random.choice(all_place_items)
    prompt = 'Pick the %s and put it on the %s'%(pick_item, place_item)
    action1 = ('pick', [pick_item])
    action2 = ('place', [place_item])
    items = [[pick_item], [place_item]]
    actions = [action1, action2]
    #print(prompt)
    #print(items)
    #print(actions)
    return prompt, items, actions

  def get_random_demo(self):
    #DEMOS = [self.generate_pick_place_demo(), \
    #         self.generate_pick_demo('block'), \
    #         self.generate_pick_demo('bowl'), \
    #         self.generate_block_diff_corner_demo(), \
    #         self.generate_block_same_corner_demo(), \
    #         self.generate_stack_demo('block'), \
    #         self.generate_stack_demo('bowl')]
    DEMOS = [self.generate_pick_place_demo(), \
             self.generate_pick_demo('block'), \
             self.generate_block_diff_corner_demo(), \
             self.generate_block_same_corner_demo(), \
             self.generate_stack_demo('block'), \
             ]
    random_demo = random.choice(DEMOS)
    return random_demo
    
  def step(self, action, target):
    if action == 'pick':
        pick_target = target
        assert(pick_target in self.PICK_TARGETS.keys())
        pick_id = self.env.obj_name_to_id[pick_target]
        pick_pose = pybullet.getBasePositionAndOrientation(pick_id)
        pick_position = np.float32(pick_pose[0])
        act = ('pick', pick_position)
    elif action == 'place':
        place_target = target
        assert(place_target in self.PLACE_TARGETS.keys())
        if place_target in self.env.obj_name_to_id:
            place_id = self.env.obj_name_to_id[place_target]
            place_pose = pybullet.getBasePositionAndOrientation(place_id)
            place_position = np.float32(place_pose[0])
        else:
            place_position = np.float32(self.PLACE_TARGETS[place_target])
        act = ('place', place_position)
    return act

  #def step(self, pick_target, place_target):
  #  #print(f'Input: {text}')

  #  assert(pick_target in self.PICK_TARGETS.keys())
  #  assert(place_target in self.PLACE_TARGETS.keys())
  #  
  #  pick_id = self.env.obj_name_to_id[pick_target]
  #  pick_pose = pybullet.getBasePositionAndOrientation(pick_id)
  #  pick_position = np.float32(pick_pose[0])

  #  if place_target in self.env.obj_name_to_id:
  #    place_id = self.env.obj_name_to_id[place_target]
  #    place_pose = pybullet.getBasePositionAndOrientation(place_id)
  #    place_position = np.float32(place_pose[0])
  #  else:
  #    place_position = np.float32(self.PLACE_TARGETS[place_target])

  #  # Add some noise to pick and place positions.
  #  #place_position[:2] += np.random.normal(scale=0.01)
  #  act = {'pick': pick_position, 'place': place_position}
  #  return act
