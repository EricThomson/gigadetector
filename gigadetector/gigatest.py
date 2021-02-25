"""
Visual example showing how to test frozen model on 
single large image using sliding window. Saves results 
(bboxes, scores, name of filepath) in giga1_od_results.pkl 
that can later be fully processed using bb_extract.py. 

Part of gigadetector repo:
https://github.com/EricThomson/gigadetector
"""
import os
import sys
import joblib

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #set to 3 to print nothing
import tensorflow as tf
#tf.enable_eager_execution()
import numpy as np
import cv2
import logging
logging.basicConfig(level = logging.WARNING)

from object_detection.utils import label_map_util
base_path = os.path.expanduser("~") + r"/gigadetector/"
sys.path.append(base_path + r'/gigadetector/')
import utils

#%% set paths
image_dir = base_path + r'data/'
save_dir = image_dir + r'processed/'
image_path = image_dir + r'giga1.png'
model_dir = base_path + r'models/'

model_path = model_dir + r'fish_frcnn_graph.pb'
labels_path = model_dir + r'fish_classes.pbtxt'
print(f"\nBeginning analysis of {image_path}\nClick Esc over movie to halt progress.")

od_filepath = save_dir + r'giga1_od_results.pkl' #previously used  datetime.now().strftime("%Y%m%d_%H%M%S"
    
if os.path.isdir(save_dir):
    pass
else:
    os.mkdir(save_dir)

#%% set basic runtime params
win_size =  1024
step_size = win_size//2
edge_min = 29  #at edges, if moving window width or height is this size or less, discard
#verbosity: 0: show nothing just save data, 1: show progress on image
verbosity = 1
if verbosity == 0:
    logging.info("\n***Note verbosity is 0, you will only see text feedback.***\n")

#save_data: toggle for test runs
save_data = 1
COLORS =  (255, 255, 255)
num_classes = 1
min_confidence = 0.95

#%% tensorflow fix for nvidia gpu cards
from tensorflow.compat.v1 import ConfigProto
#from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
#config.gpu_options.per_process_gpu_memory_fraction = 0.9
session = tf.compat.v1.Session(config=config) #InteractiveSession(config=config)



#%% Initialize model and create context manager to load model from disk
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
categories = label_map_util.convert_label_map_to_categories(labelMap,
                                                            max_num_classes = num_classes,
                                                            use_display_name=True)
categoryIdx = label_map_util.create_category_index(categories)


#%% Load image and set up window parameters
rois_all = []
boxes_all = []
scores_all = []

image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
(h,w) = image.shape
winW = win_size
winH = win_size

try:
    assert(step_size < max([winW, winH]))
except AssertionError:
    logging.error(" Attempt to step in increment larger than sub-image window size. This won't end well.")
if winW >= w and winH >= h:
    logging.warning(" Moving window is as large as the image. This is an unuusal case.")
if winW > w:
    winW = w
if winH > h:
    winH = h
inbounds_x = True  #could rename to inbounds
inbounds_xy = True
final_plot = False


#%% Cycle through applying model to each bit
if verbosity:
    ("Running analysis. Press escape to quit.")

clone = image.copy()
line_width = clone.shape[1]//300

if verbosity:
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image', 600, 800)  #width, height
    cv2.moveWindow('image', 500, 70)  #x y pos on screen
    cv2.imshow("image", clone)
    cv2.namedWindow('col_subimage', cv2.WINDOW_NORMAL)
    cv2.moveWindow('col_subimage', 75, 70)  #x y pos on screen
with model.as_default():
    with tf.compat.v1.Session(graph=model) as sess:
        for (x, y, sub_image) in utils.sliding_window(image, stepSize = step_size, windowSize=(winW, winH)):
            # if the window does not meet our desired window size, ignore it
            #if window.shape[0] != winH or window.shape[1] != winW:
            #    continue
            logging.debug(f"inbounds_x: {inbounds_x}")
            # Check on subimage size: if smaller than min, skip
            (y_sub, x_sub) = sub_image.shape
            if y_sub < edge_min or x_sub < edge_min:
                continue

            # check to see if x was inbounds, but goes out of bounds
            if x + winW > w and inbounds_x and inbounds_xy:
                logging.debug("\tSliding window resetting x next iteration")
                inbounds_x = False  #x window has gone out of bounds, so don't go next time
                if y + winH > h and inbounds_xy:  #  y window has gone out of bounds, so maybe don't go next time (depending on x)
                    inbounds_xy = False
                    logging.debug(inbounds_xy)
                    logging.debug("xy out of bounds set")
                continue
            else:
                inbounds_x = True
            # create a session to perform inference


            # from pyimagesearch: grab a reference to the input image tensor and the boxes tensor
                # if the window does not meet our desired window size, ignore it
            imageTensor = model.get_tensor_by_name("image_tensor:0")
            boxesTensor = model.get_tensor_by_name("detection_boxes:0")

            # for each bounding box we would like to know the score (i.e., probability) and class label
            scoresTensor = model.get_tensor_by_name("detection_scores:0")
            classesTensor = model.get_tensor_by_name("detection_classes:0")
            numDetections = model.get_tensor_by_name("num_detections:0")
            if verbosity:
                k = cv2.waitKey(1)
            else:
                k = -1

            if k == 27:
                break
            else:
                current_roi = np.array([y, y+winH, x, x+winW])
                #sub_image = clone[current_roi[0]: current_roi[1], current_roi[2]: current_roi[3]]
                logging.info(f"sub_image roi: {current_roi}")

                if verbosity:
                    #for showing the moving window over the original image
                    copy_big = image.copy()
                    cv2.rectangle(copy_big, (x, y), (x + winW, y + winH), COLORS, line_width)
                    cv2.imshow("image", copy_big)

                #load subimage
                #cv2.imshow('subimage', sub_image)

                #NOW APPLY PREVIOUS STUFF TO SUB_IMAGE
                (H, W) = sub_image.shape
                logging.debug(f"Subimage shape: {H, W}")

                # prepare the image for display (output) and detection (image_color), respectively
                display_image = sub_image.copy()
                image_color = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB)
                image_color = np.expand_dims(image_color, axis=0)

                # perform inference and compute the bounding boxes,  probabilities, and class labels
                (boxes, scores, labels, N) = sess.run([boxesTensor, scoresTensor, classesTensor, numDetections],
                                                       feed_dict = {imageTensor: image_color})

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

                    logging.info(f"Box: {box}")
                    logging.info(f"Score: {score}")
                    # scale the bounding box from the range [0, 1] to [W, H]
                    (startY, startX, endY, endX) = box
                    startX = int(startX * W)
                    startY = int(startY * H)
                    endX = int(endX * W)
                    endY = int(endY * H)

                    #Note this is in
                    tmp_box = np.array([startX+current_roi[2], startY+current_roi[0],
                                                 endX+current_roi[2], endY+current_roi[0]])

                    above_thresh_boxes.append(tmp_box)
                    above_thresh_scores.append(score)
                    if verbosity:
                        # draw the prediction on the output image
                        label = categoryIdx[label]
                        idx = int(label["id"]) - 1
                        label = "{}: {:.2f}".format(label["name"], score)
                        cv2.rectangle(display_image, (startX, startY), (endX, endY), COLORS, 3)
                        y = startY - 10 if startY - 10 > 10 else startY + 10
                        cv2.putText(display_image, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, COLORS, 3)


            #This is for sliding window
            if final_plot:
                logging.debug("Final plot, breaking")
                break
            if not inbounds_xy:
                logging.debug("Setting final plot to true")
                final_plot = True

            boxes_all.append(above_thresh_boxes)
            scores_all.append(above_thresh_scores)
            rois_all.append(current_roi)

            # show the fully formatted output image
            if verbosity:
                cv2.imshow("col_subimage", display_image)
                #time.sleep(0.1)

print("Analysis has finished: click escape to close window.")
k = cv2.waitKey()
if k == 27:
    cv2.destroyAllWindows()

logging.debug(f"bounding boxes: {boxes_all}")
logging.debug(f"scores: {scores_all}")


#%%
if save_data:
    logging.info(f"Saving data to {od_filepath}")
    data_to_save = {'bboxes': boxes_all, 'scores': scores_all, 'rois': rois_all, 'fname': image_path}
    with open(od_filepath, 'wb') as fp:
        joblib.dump(data_to_save, fp)
    print(f"gigadetector test finished successfully!\ndata saved to {od_filepath}")

#%% Next, to process this go to bb_draw.py














# :)
