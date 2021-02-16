# USAGE (at command line)
# python predict_working.py 
# --model /home/eric/deep_learning/fish/experiments/training/exported_model/frozen_inference_graph.pb 
# --labels /home/eric/deep_learning/fish/records/classes.pbtxt 
# --image /home/eric/deep_learning/fish/data/test_data/019_20191021_095624.bmp 
# --num-classes 1

# import the necessary packages
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #set to 3 to print nothing
from object_detection.utils import label_map_util
import tensorflow as tf
import numpy as np
import argparse
import cv2

#import tensorflow.contrib.slim as slim
#%% tensorflow fix
from tensorflow.compat.v1 import ConfigProto
#from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
#config.gpu_options.per_process_gpu_memory_fraction = 0.9
session = tf.Session(config=config) #InteractiveSession(config=config)

#%%
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="base path for frozen checkpoint detection graph")
ap.add_argument("-l", "--labels", required=True,
	help="labels file")
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
ap.add_argument("-n", "--num-classes", type=int, required=True,
	help="# of class labels")
ap.add_argument("-c", "--min-confidence", type=float, default=0.5,
	help="minimum probability used to filter weak detections")
args = vars(ap.parse_args())

# initialize a set of colors for our class labels
#COLORS = np.random.uniform(0, 255, size=(args["num_classes"], 3))
#COLORS = np.random.uniform(0, 255, size=(1, 3))
#COLORS = np.array([255, 255, 0], dtype = np.float64).reshape((1,3))
COLORS =  (0, 255, 255)

# initialize the model
model = tf.Graph()

# create a context manager that makes this model the default one for
# execution
with model.as_default():
	# initialize the graph definition
	graphDef = tf.GraphDef()

	# load the graph from disk
	with tf.gfile.GFile(args["model"], "rb") as f:
		serializedGraph = f.read()
		graphDef.ParseFromString(serializedGraph)
		tf.import_graph_def(graphDef, name="")

# load the class labels from disk
labelMap = label_map_util.load_labelmap(args["labels"])
categories = label_map_util.convert_label_map_to_categories(labelMap, max_num_classes=args["num_classes"],
	                                                        use_display_name=True)
categoryIdx = label_map_util.create_category_index(categories)

# create a session to perform inference
with model.as_default():
	with tf.Session(graph=model) as sess:
		# grab a reference to the input image tensor and the boxes tensor
		imageTensor = model.get_tensor_by_name("image_tensor:0")
		boxesTensor = model.get_tensor_by_name("detection_boxes:0")

		# for each bounding box we would like to know the score (i.e., probability) and class label
		scoresTensor = model.get_tensor_by_name("detection_scores:0")
		classesTensor = model.get_tensor_by_name("detection_classes:0")
		numDetections = model.get_tensor_by_name("num_detections:0")

		# load the image from disk
		image = cv2.imread(args["image"])
		(H, W) = image.shape[:2]

		# prepare the image for detection
		(H, W) = image.shape[:2]
		output = image.copy()
		image = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
		image = np.expand_dims(image, axis=0)

		# perform inference and compute the bounding boxes,  probabilities, and class labels
		(boxes, scores, labels, N) = sess.run([boxesTensor, scoresTensor, classesTensor, numDetections],
			                                   feed_dict={imageTensor: image})

		# squeeze the lists into a single dimension
		boxes = np.squeeze(boxes)
		scores = np.squeeze(scores)
		labels = np.squeeze(labels)

		# loop over the bounding box predictions and draw them into the image
		for (box, score, label) in zip(boxes, scores, labels):
			# if the predicted probability is less than the minimum confidence, ignore it
			if score < args["min_confidence"]:
				#print("Prediction less than min_confidence, skipping")
				continue

			# scale the bounding box from the range [0, 1] to [W, H]
			(startY, startX, endY, endX) = box
			startX = int(startX * W)
			startY = int(startY * H)
			endX = int(endX * W)
			endY = int(endY * H)

			# draw the prediction on the output image
			label = categoryIdx[label]
			idx = int(label["id"]) - 1
			label = "{}: {:.2f}".format(label["name"], score)
			cv2.rectangle(output, (startX, startY), (endX, endY), COLORS, 4)
			y = startY - 10 if startY - 10 > 10 else startY + 10
			cv2.putText(output, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 1.2, COLORS, thickness =  3)

            
		# show the fully formatted output image
		cv2.namedWindow('Output', cv2.WINDOW_NORMAL)
		cv2.imshow("Output", output)
		cv2.resizeWindow("Output", 800, 800)
		cv2.moveWindow("Output", 200, 100)  #x y pos on screen
		cv2.waitKey(0)  #wait for 0 key to be pressed (or escape key)
