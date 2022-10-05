import cv2
import os
import numpy as np

def vis(img, keypoints, text):
    text = str(text)
    for kpt in keypoints:
        #cv2.circle(img, tuple(kpt), 4, (0,160,255), -1)
        cv2.circle(img, tuple(kpt), 4, (255,255,255), -1)
    cv2.putText(img, text, (10,10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,255), 1, 2)
    #cv2.imshow('img', img)
    #cv2.waitKey(0)
    return img

if __name__ == '__main__':
    dset_dir = 'dset'
    image_dir = os.path.join(dset_dir, 'images')
    keypoints_dir = os.path.join(dset_dir, 'keypoints')
    lang_dir = os.path.join(dset_dir, 'lang')
    output_dir = 'vis'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    for i, fn in enumerate(sorted(os.listdir(image_dir))):
        img = cv2.imread(os.path.join(image_dir, fn))
        H,W,C = img.shape
        keypoints = np.load(os.path.join(keypoints_dir, '%05d.npy'%i)).astype(int)
        text = np.load(os.path.join(lang_dir, '%05d.npy'%i))
        visualization = vis(img, keypoints, text)
        cv2.imwrite('%s/%05d.jpg'%(output_dir,i), visualization)
    
