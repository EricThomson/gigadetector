#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TO DO:
    run and document gigaviewer (maybe get the gigaviewer from windows machine it is
                                 a bit better written/documented)
    Document all this stuff at the readme
    Get the big stuff hosted check that it works on another machine.
    Let mark/colin know


Process bounding boxes on multiple images analyzed using predict_folder (which did
the sliding window analysis). This is basically bb_draw wrapped around processed data.

Part of gigadetector repo:
https://github.com/EricThomson/gigadetector
"""
import os
import sys
import cv2
import numpy as np
import joblib
import logging
logging.basicConfig(level = logging.DEBUG)

base_path = os.path.expanduser("~") + r"/gigadetector/"
sys.path.append(base_path + r'/gigadetector/')
import utils

#%% set parameters
max_num_objects =100# will winnow down to this
conf_threshold = 0.9  #filter out boxes below this
nms_threshold = 0.5    # with overlap equal to/greater than this, include it for nms
overlap_threshold = 90 #overlap filter (for those nms doesn't get...overlap to the rescue)
area_threshold = 12_000
debug = 0  #display images (not really useful for full folder)
save = 2  #0 do not, 1: save bboxes etc, 2: also save images with bboxes drawn on them

#%% load data
bb_filename = r'gigafolder_od_results.pkl'
analysis_path = base_path + r'data/processed/'
bb_filepath = analysis_path + bb_filename
save_path = analysis_path

bb_color = (255, 255, 255)

#following contains 'data_paths' (pkl of bboxes, scores, etc for each image)
# as well as 'image-paths' (paths to actual images)
saved_paths = joblib.load(bb_filepath)
data_paths = saved_paths['data_paths']
image_paths = saved_paths['image_paths']
num_paths = len(data_paths)
print(f"There are {num_paths} paths")

if os.path.isdir(save_path):
    pass
else:
    os.mkdir(save_path)

all_bboxes = []
all_scores = []
all_areas = []
all_filepaths = []

assert len(image_paths) == len(data_paths), print("different number of data and image")
#%%
for current_ind in range(num_paths):
    print(f"{current_ind+1}/{num_paths}")
    bb_filepath = data_paths[current_ind]
    image_path = image_paths[current_ind]
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    image_save_path = save_path + image_name + r'_processed.png'

    print(f"\tWorking on {image_name}")
    saved_data = joblib.load(bb_filepath)
    bboxes = saved_data['bboxes']
    scores = saved_data['scores']
    rois = saved_data['rois']
    num_rois = len(rois)

    #% Filter out empty boxes yielding *_initial estimates
    bboxes_initial, scores_initial = utils.bb_filter_initial(bboxes, scores)
    num_initial = len(scores_initial)
    bboxes_initial_nms = utils.bbox_std2ocv(bboxes_initial)

    #% Filter using non-max suppression yielding bboxes_nms_filtered
    box_inds_nms = list(cv2.dnn.NMSBoxes(bboxes_initial_nms, scores_initial, conf_threshold, nms_threshold).reshape(-1,))
    bboxes_nms_filtered = [bboxes_initial[ind] for ind in box_inds_nms]
    scores_nms_filtered = [scores_initial[ind] for ind in box_inds_nms]
    num_nms_filtered = len(scores_nms_filtered)

    #% Get bbox areas (will save these for later so do for all initial boxes)
    areas_nms_filtered = []
    for bbox in bboxes_nms_filtered:
        areas_nms_filtered.append(utils.bb_area(np.asarray(bbox)))


    #%  Filter out the little guys
    bboxes_area_filtered = bboxes_nms_filtered
    scores_area_filtered = scores_nms_filtered
    areas_area_filtered = areas_nms_filtered
    for ind, area in enumerate(areas_nms_filtered):
        if area < area_threshold:
            bboxes_area_filtered = np.delete(bboxes_area_filtered, ind, axis = 0)
            scores_area_filtered = np.delete(scores_area_filtered, ind)
            areas_area_filtered = np.delete(bboxes_area_filtered, ind)
    num_area_filtered = len(scores_area_filtered)

    #% Percent overlap filter
    overlaps = utils.pairwise_overlap(bboxes_area_filtered)
    inds_to_remove = utils.overlap_suppression(overlaps, overlap_threshold)
    bboxes_overlap_filtered = bboxes_area_filtered
    scores_overlap_filtered = scores_area_filtered
    areas_overlap_filtered = areas_area_filtered
    if inds_to_remove:
        bboxes_overlap_filtered = np.delete(bboxes_overlap_filtered, inds_to_remove, axis = 0)
        scores_overlap_filtered = np.delete(scores_overlap_filtered, inds_to_remove)
        areas_overlap_filtered = np.delete(areas_overlap_filtered, inds_to_remove)

    #That's all she wrote: add these go the list of boxes, scores, areas
    all_bboxes.append(bboxes_overlap_filtered)
    all_scores.append(scores_overlap_filtered)
    all_areas.append(areas_overlap_filtered)
    all_filepaths.append(image_path)


    #% If image save ==2, draw final bbox_overlap filtered boxes into the image and save that shit
    #% save image/data
    if save == 2:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image = utils.put_bboxes_scores(image,
                         bboxes_overlap_filtered,
                         scores_overlap_filtered,
                         bb_color = (255, 255, 255),
                         line_width = 10,
                         text_thickness = 3)

        save_fname = save_path + r'processed_images/' + image_name + '_bbs.png'
        print(f"\tSaving to {save_fname}")
        cv2.imwrite(save_fname, image, [cv2.IMWRITE_PNG_COMPRESSION, 2])

        # To see it -- this is really for debugging not testing in full loop (yet)
        if debug:
            utils.cv_imshow(image, name = "filtered", xy = (2300, 50), shape = (1000, 1000))

#%%
if save:
    full_od_data = {'all_bboxes': all_bboxes,
                 'all_scores': all_scores,
                 'all_areas': all_areas,
                 'all_filepaths': all_filepaths}
    full_data_path = save_path + bb_filename
    with open(full_data_path, 'wb') as fp:
        joblib.dump(full_od_data, fp)
        print(f"DONE!!!\nBox/score/area data/path saved to\n{full_data_path}")
