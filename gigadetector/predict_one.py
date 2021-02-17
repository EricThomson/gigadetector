"""
Test model on a single small image with two fish.
Code partly adapted from od api and pyimagesearch.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #set to 3 to print nothing
from object_detection.utils import label_map_util
import tensorflow as tf
import numpy as np
import cv2

#%% tensorflow fix for rtx/gtx
from tensorflow.compat.v1 import ConfigProto
config = ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config) #InteractiveSession(config=config)

#%% set paths
base_path = os.path.expanduser("~") + r"/gigadetector/"
image_dir = base_path + r'test_data/'
image_path = image_dir + 'fish.png'
model_dir = base_path + r'models/'
model_path = model_dir + r'fish_frcnn_graph.pb'
labels_path = model_dir + r'fish_classes.pbtxt'
label_color =  (0, 255, 255)
num_classes = 1
min_confidence = 0.9
print(f"\nSetting up analysis of {image_path}")

#%% initialize and run the model
model = tf.Graph()
with model.as_default():
	# initialize the graph definition
	graphDef = tf.GraphDef()
	# load the graph from disk
	with tf.gfile.GFile(model_path, "rb") as f:
		serializedGraph = f.read()
		graphDef.ParseFromString(serializedGraph)
		tf.import_graph_def(graphDef, name="")

# load the class labels from disk
labelMap = label_map_util.load_labelmap(labels_path)
categories = label_map_util.convert_label_map_to_categories(labelMap, max_num_classes=num_classes,
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
		image = cv2.imread(image_path)
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
			if score < min_confidence:
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
			cv2.rectangle(output, (startX, startY), (endX, endY), label_color, 4)
			y = startY - 10 if startY - 10 > 10 else startY + 10
			cv2.putText(output, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 
                        1.2, label_color, thickness =  3)

            
#%% show the fully formatted output image
print("Final analyzed image should be showing. Press ESC to close.")
cv2.namedWindow('Test Output', cv2.WINDOW_NORMAL)
cv2.imshow("Test Output", output)
cv2.resizeWindow("Test Output", 800, 800)
cv2.moveWindow("Test Output", 200, 100)  #x y pos on screen
k = cv2.waitKey()
if k == 27:
    cv2.destroyAllWindows()
