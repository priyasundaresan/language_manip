import numpy as np
import random
import cv2
import matplotlib.pyplot as plt
from moviepy.editor import ImageSequenceClip
import threading
import os
import time
import pickle
from PIL import Image
import pybullet
import pybullet_data
from robotiq import Robotiq2F85
from pick_place import PickPlaceEnv
#from scripted_policy_paired import ScriptedPolicy
from scripted_policy_multistep import ScriptedPolicy

PICK_TARGETS = {
  "blue block": None,
  "red block": None,
  "green block": None,
  "yellow block": None,
  "blue bowl": None,
  "red bowl": None,
  "green bowl": None,
  "yellow bowl": None,
}

COLORS = {
    "blue":   (78/255,  121/255, 167/255, 255/255),
    "red":    (255/255,  87/255,  89/255, 255/255),
    "green":  (89/255,  169/255,  79/255, 255/255),
    "yellow": (237/255, 201/255,  72/255, 255/255),
}

PLACE_TARGETS = {
  "blue block": None,
  "red block": None,
  "green block": None,
  "yellow block": None,

  "blue bowl": None,
  "red bowl": None,
  "green bowl": None,
  "yellow bowl": None,

  "top left corner":     (-0.3 + 0.05, -0.2 - 0.05, 0),
  "top right corner":    (0.3 - 0.05,  -0.2 - 0.05, 0),
  "middle":              (0,           -0.5,        0),
  "bottom left corner":  (-0.3 + 0.05, -0.8 + 0.05, 0),
  "bottom right corner": (0.3 - 0.05,  -0.8 + 0.05, 0),
}

#PIXEL_SIZE = 0.00267857
PIXEL_SIZE = 0.0025
BOUNDS = np.float32([[-0.3, 0.3], [-0.8, -0.2], [0, 0.15]])  # X Y Z

#@markdown Gripper (Robotiq 2F85) code

def xyz_to_pix(position):
  """Convert from 3D position to pixel location on heightmap."""
  u = int(np.round((BOUNDS[1, 1] - position[1]) / PIXEL_SIZE))
  v = int(np.round((position[0] - BOUNDS[0, 0]) / PIXEL_SIZE))
  return (u, v)

if 'env' in locals():
  # Safely exit gripper threading before re-initializing environment.
  env.gripper.running = False
  while env.gripper.constraints_thread.isAlive():
    time.sleep(0.01)
env = PickPlaceEnv(BOUNDS, COLORS, PIXEL_SIZE)

# Define and reset environment.
config = {'pick':  PICK_TARGETS.keys(),
          'place': PLACE_TARGETS.keys()}
np.random.seed(42)
obs = env.reset(config)

# MAKE A NEW DATASET
output_gifs_dir = 'gifs'
if not os.path.exists(output_gifs_dir):
    os.mkdir(output_gifs_dir)


dset_dir = 'dset'
images_dir = os.path.join(dset_dir, 'images')
lang_dir = os.path.join(dset_dir, 'lang')
keypoints_dir = os.path.join(dset_dir, 'keypoints')
for d in [dset_dir, images_dir, lang_dir, keypoints_dir]:
    if not os.path.exists(d):
        os.mkdir(d)

policy = ScriptedPolicy(env, PICK_TARGETS, PLACE_TARGETS)

image_size = (240, 240)
#intrinsics = (120., 0, 120., 0, 120., 120., 0, 0, 1)
intrinsics = (190., 0, 190., 0, 190., 190., 0, 0, 1)
fmat = env.get_fmat(image_size, intrinsics)
#fmat = env.get_fmat()

data_idx = 0
MAX_DEMOS = 350
for i in range(MAX_DEMOS):
    #prompt, items_all, actions = policy.generate_stack_demo('block')
    np.random.seed(data_idx)
    obs = env.reset(config)
    prompt, items_all, actions = policy.get_random_demo()
    #print('here', result, type(result))
    prompt = random.choice(policy.prompt_augs(prompt))
    for (items, (action, target)) in zip(items_all, actions):
        print(data_idx)
        
        target = target[0]
        act = policy.step(action, target)
        action_type, xyz_coord = act

        img = env.get_camera_image()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        #img = env.get_camera_image_top()
        #img = np.flipud(img.transpose(1, 0, 2))
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        pixels = []
        for item in items:
            if item in env.obj_name_to_id:
                item_id = env.obj_name_to_id[item]
                pose = pybullet.getBasePositionAndOrientation(item_id)
                position = np.float32(pose[0])
            else:
                position = np.float32(PLACE_TARGETS[item])

            pixel = (fmat @ np.hstack((position, [1])))[:3]*np.array([240, 240, 0]) + np.array([120, 120, 0])
            pixel = pixel[:-1].astype(int)
            pixel[1] = 240 - pixel[1]
            pixels.append(pixel)
            #pixels.append(xyz_to_pix(position)[::-1])

        np.save('%s/%05d.npy'%(lang_dir, data_idx), prompt)
        np.save('%s/%05d.npy'%(keypoints_dir, data_idx), pixels)
        cv2.imwrite('%s/%05d.jpg'%(images_dir, data_idx), img)

        if action_type == 'pick':
            obs, _, _, _ = env.step_pick(xyz_coord)
        else:
            obs, _, _, _ = env.step_place(xyz_coord)

        data_idx += 1
