import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

class ObjectDetector():
	def __init__(self,path_to_labels,path_to_ckpt,num_classes):
		self.PATH_TO_LABELS = path_to_labels
		self.PATH_TO_CKPT = path_to_ckpt
		self.NUM_CLASSES = num_classes

	def load_model(self):
		self.detection_graph = tf.Graph()
		with self.detection_graph.as_default():
			od_graph_def = tf.GraphDef()
			with tf.gfile.GFile(self.PATH_TO_CKPT, 'rb') as fid:
				serialized_graph = fid.read()
				od_graph_def.ParseFromString(serialized_graph)
				tf.import_graph_def(od_graph_def, name='')

			self.sess = tf.Session(graph=self.detection_graph)


	def detected_setting(self):
		self.load_model()

		# Input tensor is the image
		self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')

		# Output tensors are the detection boxes, scores, and classes
		# Each box represents a part of the image where a particular object was detected
		self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')

		# Each score represents level of confidence for each of the objects.
		# The score is shown on the result image, together with the class label.
		self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
		self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')

		# Number of objects detected
		self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')


	def get_detected_info(self,path_to_image,category_index):

		self.detected_setting()

		# Load image using OpenCV and
		# expand image dimensions to have shape: [1, None, None, 3]
		# i.e. a single-column array, where each item in the column has the pixel RGB value
		image = cv2.imread(path_to_image)
		image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		image_expanded = np.expand_dims(image_rgb, axis=0)

		# Perform the actual detection by running the model with the image as input
		(boxes, scores, classes, num) = self.sess.run([self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],\
													  feed_dict={self.image_tensor: image_expanded})

		# Draw the results of the detection (aka 'visulaize the results')
		vis_util.visualize_boxes_and_labels_on_image_array(
			image_rgb,
			np.squeeze(boxes),
			np.squeeze(classes).astype(np.int32),
			np.squeeze(scores),
			category_index,
			use_normalized_coordinates=True,
			line_thickness=4,
			min_score_thresh=0.80)

		# All the results have been drawn on image. Now display the image.
		plt.figure(figsize=(20,20))
		plt.imshow(image_rgb)
		plt.show()
