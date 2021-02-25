"""
Example pipeline for extracting unique bboxes from the set of all bboxes pulled
in gigatest.py. Saves final estimate as giga1_boxes.png.

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

bb_color = (255, 255, 255)
# Useful constraint: can go below this, but if you go above it, it will try to find boxes to kill.
max_num_objects = 100
area_threshold = 12_000  #fish should be above this
conf_threshold = 0.95  #filter out boxes below this
nms_threshold = 0.5    # with overlap equal to/greater than this, include in nms competition
overlap_threshold = 85 #for overlap trimming step, this is the threshold for equality
save = 1
verbose = 0  #to draw intermediate images -- used when stepping through and debugging
#load data
od_filename = r'giga1_od_results.pkl'
analysis_path = base_path + r'data/processed/'
od_filepath = analysis_path + od_filename
processed_image_path = analysis_path + 'processed_images/'

if os.path.isdir(processed_image_path):
    pass
else:
    os.mkdir(processed_image_path)

#%%
with open(od_filepath, 'rb') as fp:
   saved_data = joblib.load(fp)

image_path = saved_data['fname']
bboxes = saved_data['bboxes']
scores = saved_data['scores']
rois = saved_data['rois']
num_rois = len(rois)
image_name = os.path.splitext(os.path.basename(image_path))[0]

print(f"Rendering bounding boxes for data from {image_path}")
print("When done inspecting image, click ESC to close window and save data.")


"""
#%% set paths
image_dir = base_path + r'data/'
save_dir = image_dir + r'processed/'
image_path = image_dir + r'giga1.png'
model_dir = base_path + r'models/'

model_path = model_dir + r'fish_frcnn_graph.pb'
labels_path = model_dir + r'fish_classes.pbtxt'
print(f"\nBeginning analysis of {image_path}\nClick Esc over movie to halt progress.")

bb_filepath = save_dir + r'giga1_od_results.pkl' #previously used  datetime.now().strftime("%Y%m%d_%H%M%S"
    
if os.path.isdir(save_dir):
    pass
else:
    os.mkdir(save_dir)
"""

#%% Extract bboxes (note they are in std format: xs, ys, xe, ye)
bboxes_initial, scores_initial = utils.bb_filter_initial(bboxes, scores)
num_initial = len(scores_initial)
bboxes_initial_nms = utils.bbox_std2ocv(bboxes_initial)

#%% to draw just the image
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
if verbose: utils.cv_imshow(image, xy = (1300, 50), shape = (1000, 1000))

#%% draw bboxes and scores
if verbose:
    utils.draw_bboxes_scores(image.copy(), bboxes_initial, scores_initial, bb_color = (255, 255, 255),
                           name = 'initial', line_width = 10, text_thickness = 3,
                           shape = (1000, 1000), xy = (1300, 50))


#%% Filter using non-max suppression yielding bboxes_nms_filtered
box_inds_nms = list(cv2.dnn.NMSBoxes(bboxes_initial_nms, scores_initial, conf_threshold, nms_threshold).reshape(-1,))
bboxes_nms_filtered = [bboxes_initial[ind] for ind in box_inds_nms]
scores_nms_filtered = [scores_initial[ind] for ind in box_inds_nms]
num_nms_filtered = len(scores_nms_filtered)

#%% Draw fish that have been nms-filtered
if verbose:
    utils.draw_bboxes_scores(image.copy(),
                             bboxes_nms_filtered,
                             scores_nms_filtered,
                             bb_color = (255, 255, 255),
                             name = 'nms filtered',
                             line_width = 10,
                             text_thickness = 3,
                             shape = (1000, 1000),
                             xy = (1300, 50))



#%% Get bbox areas (will save these for later so do for all initial boxes)
areas_nms_filtered = []
for bbox in bboxes_nms_filtered:
    areas_nms_filtered.append(utils.bb_area(np.asarray(bbox)))


#%% Filter out the little guys, if this was desired
bboxes_area_filtered = bboxes_nms_filtered
scores_area_filtered = scores_nms_filtered
areas_area_filtered = areas_nms_filtered

for ind, area in enumerate(areas_nms_filtered):
    if area < area_threshold:
        bboxes_area_filtered = np.delete(bboxes_area_filtered, ind, axis = 0)
        scores_area_filtered = np.delete(scores_area_filtered, ind)
        areas_area_filtered = np.delete(areas_area_filtered, ind)
num_area_filtered = len(scores_area_filtered)

#%% To view area-filtered bboxes
if verbose:
    utils.draw_bboxes_scores(image.copy(),
                             bboxes_area_filtered,
                             scores_area_filtered,
                             bb_color = (255, 255, 255),
                             name = 'area filtered',
                             line_width = 10,
                             text_thickness = 3,
                             shape = (1000, 1000),
                             xy = (1300, 50))



#%% Overlap filter:
# calculate percent overlap: overlap(AB) is percentage of A that is in B.
# It is asymmetric and can be used to filter out some BBs that were missed by nms
overlaps = utils.pairwise_overlap(bboxes_area_filtered)
inds_to_remove = utils.overlap_suppression(overlaps, overlap_threshold)
bboxes_overlap_filtered = bboxes_area_filtered
scores_overlap_filtered = scores_area_filtered
areas_overlap_filtered = areas_area_filtered
if inds_to_remove:
    bboxes_overlap_filtered = np.delete(bboxes_overlap_filtered, inds_to_remove, axis = 0)
    scores_overlap_filtered = np.delete(scores_overlap_filtered, inds_to_remove)
    areas_overlap_filtered = np.delete(areas_overlap_filtered, inds_to_remove)

#%% Draw the final image
utils.draw_bboxes_scores(image,
                         bboxes_overlap_filtered,
                         scores_overlap_filtered,
                         bb_color = (255, 255, 255),
                         name = 'overlap filtered',
                         line_width = 10,
                         text_thickness = 3,
                         shape = (950, 950),
                         xy = (1300, 50))


#%%  Save the image with bounding boxes if you want
# should also save bboxes, confidence, etc. (see ): see bb_analysis_folder.py for this
if save:
    save_fname = processed_image_path + 'giga1_bboxes.png'
    print(f"gigadetector saving {save_fname}")
    # saves without compression
    a = cv2.imwrite(save_fname, image, [cv2.IMWRITE_PNG_COMPRESSION, 2])
    if not a:
        print(f"{save_fname} did not save")
    else:
        print("gigadetector test finished successfully saving!")
