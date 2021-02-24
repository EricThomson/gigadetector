#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Draw boxes on images processed using gigadetector pipeline.

click n to keep going, escape to stop.
If you press q I'm not sure what will happen

"""
# Import stuff
import sys
import os
import joblib
import cv2

base_path = os.path.expanduser("~") + r"/gigadetector/"
sys.path.append(base_path + r'/gigadetector/')
import utils

#%% set path to final results file, and load data
# includes bboxes, scores, areas, and image paths
# note image paths might change if someone moves images but final node in path
# shouldn't.
processed_image_folder = base_path + r'data/processed/'

# Final bbox and confidence output of faster-rcnn + bbox trimming (bb_analysis_folder.py)
results_file = r'gigafolder_bb_results.pkl'  #1801-2648
results_path = processed_image_folder + results_file

with open(results_path, 'rb') as f:
    analysis_data = joblib.load(results_path)

#%% Extract it all
all_bboxes = analysis_data['all_bboxes']
all_scores = analysis_data['all_scores']
all_areas = analysis_data['all_areas']
image_paths = analysis_data['all_filepaths']
num_images = len(image_paths)
print(f"There are {num_images} images for which you have detection data.")
print(image_paths)


#%% optional test case
"""
OPTIONAL -- uncomment following to run
This is to run on a single image just to make sure it works for one image
"""
# print("\ngigaviewer Tester\nClick escape to break out, n to move on to next image.\n")
# image_ind = 1
# bboxes = all_bboxes[image_ind]
# scores = all_scores[image_ind]
# image_path = image_paths[image_ind]
# image = cv2.imread(image_path)
# utils.draw_bboxes_scores(image.copy(), bboxes, scores, bb_color = (255, 255, 255),
#                       name = 'ViewTester', line_width = 10, text_thickness = 3,
#                       shape = (900, 1000), xy = (130, 50))



#%% If test case seems ok, start from ind you want, and cycle through images
print("\ngigaimage inspector\nClick escape to break out, n to move on to next image.\n")
start_image_ind = 0
window_open = False
for ind in range(start_image_ind, num_images):
    print(f"Working on image {ind} out of {num_images-1}")
    bboxes = all_bboxes[ind]
    scores = all_scores[ind]
    image_path = image_paths[ind]
    print(f"\tLoading{image_path}")
    boxed_image = utils.put_bboxes_scores(cv2.imread(image_path), bboxes, scores,
                                          bb_color = (255, 255, 255),
                                          line_width = 10, text_thickness = 3)
    if window_open:
        cv2.destroyWindow(str(ind-1))
    else:
        window_open = True
    utils.cv_loopshow(boxed_image,
                      name = str(ind),
                      shape = (950, 950),
                      xy = (130, 40))

    k = cv2.waitKey()
    if k == 27:
        break
    elif k == ord('n'):
        continue

cv2.destroyAllWindows()

print("\nDONE!!!")
