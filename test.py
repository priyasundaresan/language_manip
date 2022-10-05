import numpy as np
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
from scripted_policy import ScriptedPolicy

PICK_TARGETS = {
  "blue block": None,
  "red block": None,
  "green block": None,
  "yellow block": None,
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

PIXEL_SIZE = 0.00267857
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
config = {'pick':  ['yellow block', 'green block', 'blue block'],
          'place': ['yellow bowl', 'green bowl', 'blue bowl']}

np.random.seed(42)
obs = env.reset(config)

plt.subplot(1, 2, 1)
img = env.get_camera_image()
plt.title('Perspective side-view')
plt.imshow(img)
plt.subplot(1, 2, 2)
img = env.get_camera_image_top()
img = np.flipud(img.transpose(1, 0, 2))
plt.title('Orthographic top-view')
plt.imshow(img)
plt.show()

# Note: orthographic cameras do not exist. But we can approximate them by
# projecting a 3D point cloud from an RGB-D camera, then unprojecting that onto
# an orthographic plane. Orthographic views are useful for spatial action maps.
plt.title('Unprojected orthographic top-view')
plt.imshow(obs['image'])
plt.show()


# MAKE A NEW DATASET
output_gifs_dir = 'gifs'
if not os.path.exists(output_gifs_dir):
    os.mkdir(output_gifs_dir)

dataset = {}
dataset_size = 10  # Size of new dataset.
dataset['image'] = np.zeros((dataset_size, 224, 224, 3), dtype=np.uint8)
dataset['pick_yx'] = np.zeros((dataset_size, 2), dtype=np.int32)
dataset['place_yx'] = np.zeros((dataset_size, 2), dtype=np.int32)
dataset['text'] = []
policy = ScriptedPolicy(env, PICK_TARGETS, PLACE_TARGETS)
data_idx = 0

while data_idx < dataset_size:
  np.random.seed(data_idx)
  num_pick, num_place = 3, 3

  # Select random objects for data collection.
  pick_items = list(PICK_TARGETS.keys())
  pick_items = np.random.choice(pick_items, size=num_pick, replace=False)
  place_items = list(PLACE_TARGETS.keys())
  for pick_item in pick_items:  # For simplicity: place items != pick items.
    place_items.remove(pick_item)
  place_items = np.random.choice(place_items, size=num_place, replace=False)
  config = {'pick': pick_items, 'place': place_items}

  # Initialize environment with selected objects.
  obs = env.reset(config)

  # Create text prompts.
  prompts = []
  for i in range(len(pick_items)):
    pick_item = pick_items[i]
    place_item = place_items[i]
    prompts.append(f'Pick the {pick_item} and place it on the {place_item}.')

  # Execute 3 pick and place actions.
  for prompt in prompts:
    act = policy.step(prompt, obs)
    dataset['text'].append(prompt)
    dataset['image'][data_idx, ...] = obs['image'].copy()
    dataset['pick_yx'][data_idx, ...] = xyz_to_pix(act['pick'])
    dataset['place_yx'][data_idx, ...] = xyz_to_pix(act['place'])
    data_idx += 1
    obs, _, _, _ = env.step(act)
    debug_clip = ImageSequenceClip(env.cache_video, fps=25)
    debug_clip.write_gif("%s/%s.gif"%(output_gifs_dir, prompt.replace(' ', '_').replace('.', '').lower()),fps=15)
    #display(debug_clip.ipython_display(autoplay=1, loop=1))
    env.cache_video = []
    if data_idx >= dataset_size:
      break

pickle.dump(dataset, open(f'dataset-{dataset_size}.pkl', 'wb'))
