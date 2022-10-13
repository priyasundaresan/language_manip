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

  def generate_all_block_demos(self):
    prompts = []
    items_to_annotate = []
    annotations = []

    prompts_all = ['Stack the blocks', \
                  'Put the blocks in different corners', \
                  'Put the blocks in the same corner', \
                  'Pick up a block']
    for prompt in prompts_all:
        blocks = [k for k in self.PICK_TARGETS.keys() if 'block' in k]
        prompt_aug = self.prompt_augs(prompt)
        annots = []
        for i, target in enumerate(blocks):
            pick_id = self.env.obj_name_to_id[target]
            pick_pose = pybullet.getBasePositionAndOrientation(pick_id)
            pick_position = np.float32(pick_pose[0])
            annots.append(pick_position)
        #annotations.append(annots)

        for p in prompt_aug:
            prompts.append(p)
            items_to_annotate.append(blocks)
            annotations.append(annots)
    return prompts, items_to_annotate

  def generate_all_bowl_demos(self):
    prompts = []
    items_to_annotate = []
    annotations = []
    prompts_all = ['Stack the bowls', \
                  'Put the bowls on top of each other', \
                  'Pick up a bowl']
    for prompt in prompts_all:
        bowls = [k for k in self.PICK_TARGETS.keys() if 'bowl' in k]
        prompt_aug = self.prompt_augs(prompt)
        annots = []
        for i, target in enumerate(bowls):
            pick_id = self.env.obj_name_to_id[target]
            pick_pose = pybullet.getBasePositionAndOrientation(pick_id)
            pick_position = np.float32(pick_pose[0])
            annots.append(pick_position)
        #annotations.append(annots)

        for p in prompt_aug:
            prompts.append(p)
            items_to_annotate.append(bowls)
            annotations.append(annots)
    return prompts, items_to_annotate

  def generate_pick_place_demo(self):
    prompt = 'Pick the %s and put it on the %s'
    prompts = []
    items_to_annotate = []
    annotations = []
    for pick_item in self.PICK_TARGETS.keys():
        place_items = [k for k in self.PLACE_TARGETS.keys() if not (k == pick_item)]
        for place_item in place_items:
            prompt_completed = prompt%(pick_item, place_item)
            prompt_aug = self.prompt_augs(prompt_completed)
            items  = [pick_item]
            annots = []
            pick_id = self.env.obj_name_to_id[pick_item]
            pick_pose = pybullet.getBasePositionAndOrientation(pick_id)
            pick_position = np.float32(pick_pose[0])
            annots.append(pick_position)
            #annotations.append(annots)
            for p in prompt_aug:
                prompts.append(p)
                items_to_annotate.append(items)
                annotations.append(annots)
    #pprint.pprint(list(zip(prompts, items_to_annotate)))
    return prompts, items_to_annotate

  def step(self, pick_target, place_target):
    print(f'Input: {text}')

    assert(pick_target in self.PICK_TARGET.keys())
    assert(place_target in self.PLACE_TARGET.keys())
    
    pick_id = self.env.obj_name_to_id[pick_target]
    pick_pose = pybullet.getBasePositionAndOrientation(pick_id)
    pick_position = np.float32(pick_pose[0])

    if place_target in self.env.obj_name_to_id:
      place_id = self.env.obj_name_to_id[place_target]
      place_pose = pybullet.getBasePositionAndOrientation(place_id)
      place_position = np.float32(place_pose[0])
    else:
      place_position = np.float32(self.PLACE_TARGETS[place_target])

    # Add some noise to pick and place positions.
    #place_position[:2] += np.random.normal(scale=0.01)
    act = {'pick': pick_position, 'place': place_position}
    return act
