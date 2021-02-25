"""
Process multiple large images in data folder.

Saves results (includes boxes, confidence, image paths) to
gigadetector/data/processed/gigafolder_od_results.pkl
You can then process these results using bb_extract_folder.py.

Part of gigadetector repo:
https://github.com/EricThomson/gigadetector
"""
import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #set to 3 to print nothing
import tensorflow as tf
import numpy as np
import cv2
import joblib

from object_detection.utils import label_map_util
base_path = os.path.expanduser("~") + r"/gigadetector/"
sys.path.append(base_path + r'/gigadetector/')
import utils

import logging
logging.basicConfig(level = logging.INFO)

win_size = 1024
step_size = win_size//2
edge_min = 29  #when subimage would be smaller than this in either dimension, move on
save_data = 1 #save_data: toggle to 0 for test runs just to make sure you get no errors
verbose = 0  #0 just text, 1 to show sliding window
if verbose == 0:
    logging.info("\n***Note verbose is 0, you will only see text feedback.***\n")

#%% tensorflow fix
from tensorflow.compat.v1 import ConfigProto
#from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
#config.gpu_options.per_process_gpu_memory_fraction = 0.8
session = tf.compat.v1.Session(config=config) #InteractiveSession(config=config)



#%% set paths

images_dir = base_path + r'data/'
save_dir = images_dir + r'processed/'
image_paths = utils.extract_paths(images_dir, extension = 'png')  #bmp
num_images = len(image_paths)
print(f"Will run Faster-RCNN on the {num_images} images in {images_dir}")
model_dir = base_path + r'models/'
model_path = model_dir + r'fish_frcnn_graph.pb'
labels_path = model_dir + r'fish_classes.pbtxt'
od_filepath = save_dir +  r'gigafolder_od_results.pkl'

#initialize list where data will be saved
od_data_all = []  


if os.path.isdir(save_dir):
    pass
else:
    os.mkdir(save_dir)

num_classes = 1
min_confidence = 0.9



"""
#%% set paths
images_dir = base_path + r'data/'
save_dir = images_dir + r'processed/'
image_path = images_dir + r'giga1.png'
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

#%%
# initialize the model
print("Loading model from disk...")
model = tf.Graph()

# create a context manager that makes this model the default one for execution
with model.as_default():
    # initialize the graph definition
    graphDef = tf.compat.v1.GraphDef()

    # load the graph from disk
    with tf.io.gfile.GFile(model_path, "rb") as f:
        serializedGraph = f.read()
        graphDef.ParseFromString(serializedGraph)
        tf.import_graph_def(graphDef, name="")

# load the class labels from disk
labelMap = label_map_util.load_labelmap(labels_path)
categories = label_map_util.convert_label_map_to_categories(labelMap, max_num_classes = num_classes,
                                                            use_display_name=True)
categoryIdx = label_map_util.create_category_index(categories)




#%% create a session to perform inference
with model.as_default():
    with tf.compat.v1.Session(graph=model) as sess:

        #% run sliding window on images in image_paths
        for ind, image_path in enumerate(image_paths):
            print(f"Processing {ind}: {image_path}")
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            (H_full,W_full) = image.shape
            winW = win_size
            winH = win_size
            num_rows_approx = H_full//step_size
            num_cols_approx = W_full//step_size
            num_windows = int(num_rows_approx*num_cols_approx + num_rows_approx + num_cols_approx)

            step_num = 0
            inbounds_x = True  #could rename to inbounds
            inbounds_xy = True

            try:
                assert(step_size < max([winW, winH]))
            except AssertionError:
                logging.error(" Attempt to step in increment larger than sub-image window size. This won't end well.")
            if winW >= W_full and winH >= H_full:
                logging.warning(" Moving window is as large as the image. This is an unuusal case.")
            if winW > W_full:
                winW = W_full
            if winH > H_full:
                winH = H_full

            # grind through session with multiple windows/bboxes
            rois_image = []
            boxes_image = []
            scores_image = []
            features_image = []

            image_copy = image.copy()
            line_width = image_copy.shape[0]//200
            if verbose:
                cv2.namedWindow('image', cv2.WINDOW_NORMAL)
                cv2.resizeWindow('image', 800, 800)  #width, height
                cv2.imshow("image", image_copy)

            for (x, y, sub_image) in utils.sliding_window(image, stepSize = step_size, windowSize=(winW, winH)):
                if verbose: cv2.namedWindow('subimage', cv2.WINDOW_NORMAL)
                if step_num % 250 == 0:
                    print(f"     {ind} On window {step_num} out of about {num_windows}")
                step_num += 1
                # Check on subimage size
                (y_sub, x_sub) = sub_image.shape
                if y_sub < edge_min or x_sub < edge_min:
                    continue

                # If x/y are out of bounds, break out of this for loop
                if x + winW > W_full and inbounds_x and inbounds_xy:
                    logging.debug("\tSliding window resetting x next iteration")
                    inbounds_x = False  #x window has gone out of bounds, so don't go next time
                    if y + winH > H_full and inbounds_xy:  #  y window has gone out of bounds, so maybe don't go next time (depending on x)
                        inbounds_xy = False
                        logging.debug(inbounds_xy)
                        logging.debug("xy out of bounds set")
                        break
                    continue
                else:
                    inbounds_x = True

                imageTensor = model.get_tensor_by_name("image_tensor:0")
                boxesTensor = model.get_tensor_by_name("detection_boxes:0")
                scoresTensor = model.get_tensor_by_name("detection_scores:0")
                classesTensor = model.get_tensor_by_name("detection_classes:0")
                numDetections = model.get_tensor_by_name("num_detections:0")
                if verbose:
                    k = cv2.waitKey(1)
                else:
                    k = -1

                if k == 27:
                    cv2.destroyAllWindows()
                    break
                else:
                    current_roi = np.array([x, y, x+winH, y+winH])  #x1, y1, x2, y2
                    logging.debug(f"sub_image roi: {current_roi}")

                    if verbose:
                        #for showing the moving window over the original image
                        cv2.rectangle(image_copy,
                                      (current_roi[0], current_roi[1]),
                                      (current_roi[2], current_roi[3]),
                                      (255, 255, 255),
                                      line_width)

                    #Analyze sub_image
                    (H_sub, W_sub) = sub_image.shape

                    # prepare the image for display (output) and detection (image_color), respectively
                    sub_image_copy = sub_image.copy()
                    image_color = cv2.cvtColor(sub_image_copy, cv2.COLOR_BGR2RGB)
                    image_color = np.expand_dims(image_color, axis=0)

                    # perform inference and compute the bounding boxes,  probabilities, class labels, features, and number of detections
                    (boxes, scores, labels, N) = sess.run([boxesTensor, scoresTensor,
                                                           classesTensor, numDetections],
                                                             feed_dict={imageTensor:image_color})

                    # squeeze the lists into a single dimension
                    boxes = np.squeeze(boxes)
                    scores = np.squeeze(scores)
                    labels = np.squeeze(labels)

                    logging.debug(f"Num boxes before confidence thresholding: {len(boxes)}")
                    above_thresh_boxes = []
                    above_thresh_scores= []

                    # loop over the bounding box predictions and draw them into the image
                    for (box, score, label) in zip(boxes, scores, labels):
                        # if the predicted probability is less than the minimum confidence, ignore it
                        if score < min_confidence:
                            continue
                        #print(box)
                        logging.debug(f"Box: {box}")
                        logging.debug(f"Score: {score}")
                        # scale the bounding box from the range [0, 1] to [W_sub, H_sub]

                        (startY, startX, endY, endX) = box
                        startX = int(startX * W_sub)
                        startY = int(startY * H_sub)
                        endX = int(endX * W_sub)
                        endY = int(endY * H_sub)
                        #Reconstruct box using ROI in large image (roi is in y1, y2, x1, x2)
                        box_full = np.array([startX+current_roi[0],
                                             startY+current_roi[1],
                                             endX+current_roi[0],
                                             endY+current_roi[1]])
                        above_thresh_boxes.append(box_full)
                        above_thresh_scores.append(score)
                        if verbose:
                            ##add RECTANGLE to image copy here as well!!! should be able to use box_full and draw in black...
                            # draw the prediction on the output image
                            label = categoryIdx[label]
                            idx = int(label["id"]) - 1
                            label = "{}: {:.2f}".format(label["name"], score)
                            # label mini image
                            cv2.rectangle(sub_image_copy, (startX, startY), (endX, endY), (220, 220, 220), line_width//4)
                            y_text = startY - 10 if startY - 10 > 10 else startY + 10
                            cv2.putText(sub_image_copy, label, (startX, y_text), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (220, 220, 220), line_width//8)

                            # label full image
                            x1 = box_full[0]
                            y1 = box_full[1]
                            x2 = box_full[2]
                            y2 = box_full[3]

                            #print(x1, y1, x2, y2)
                            cv2.rectangle(image_copy,
                                          (x1, y1),
                                          (x2, y2),
                                          (200, 200, 200),
                                          line_width//2)
                            cv2.putText(image_copy, label, (x1, y1-20), cv2.FONT_HERSHEY_SIMPLEX, 2, (220, 220, 220), line_width//4)

                    if verbose:
                        cv2.imshow("subimage", sub_image_copy)
                        cv2.imshow("image", image_copy)

                boxes_image.append(above_thresh_boxes)
                scores_image.append(above_thresh_scores)
                rois_image.append(current_roi)

            # Save data for just this image so we don't lose it.
            # But also append image paths and bb data to their lists for saving at the end
            if save_data:
                data_to_save = {'bboxes': boxes_image,
                                'scores': scores_image,
                                'rois': rois_image,
                                'fname': image_path}
                od_data_all.append(data_to_save)
print("Done performing inference on images in folder")

#%% save *all* data -- bboxes and file paths that you've saved in lists
if save_data:
    print(f"Saving data and image paths to {od_filepath}")
    with open(od_filepath, 'wb') as fp:
        joblib.dump(od_data_all, fp)

#Next, to process this go to bb_analysis_folder or bb_analysis_sliding.py















# :)
