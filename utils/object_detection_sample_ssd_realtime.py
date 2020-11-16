import cv2
import numpy as np
import os, sys
from time import strftime, localtime
import random
from openvino.inference_engine import IENetwork, IEPlugin

# plugin = IEPlugin("MYRIAD", "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64")
plugin = IEPlugin("GPU", "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64")
# plugin = IEPlugin("CPU", "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64")
# plugin.add_cpu_extension("/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so")

model_xml = '/home/intel/Desktop/intel-object-detection-final/models/openvino/frozen_inference_graph.xml'
model_bin = '/home/intel/Desktop/intel-object-detection-final/models/openvino/frozen_inference_graph.bin'

print('Loading network files:\n\t{}\n\t{}'.format(model_xml, model_bin))

net = IENetwork(model=model_xml, weights=model_bin)
#print("net inputs: {}, outputs: {}".format(net.inputs['data'].shape, net.outputs['detection_out'].shape))



net.batch_size = 1

input_blob = next(iter(net.inputs))
out_blob = next(iter(net.outputs))

exec_net = plugin.load(network=net)

cap = cv2.VideoCapture(0)
if not cap.isOpened(): sys.exit('camera error')

labels = ['Abyssinian',
 'american_bulldog',
 'american_pit_bull_terrier',
 'basset_hound',
 'beagle',
 'Bengal',
 'Birman',
 'Bombay',
 'boxer',
 'British_Shorthair',

 'chihuahua',
 'Egyptian_Mau',
 'english_cocker_spaniel',
 'english_setter',
 'german_shorthaired',
 'great_pyrenees',
 'havanese',
 'japanese_chin',
 'keeshond',
 'leonberger',

 'Maine_Coon',
 'miniature_pinscher',
 'newfoundland',
 'Persian',
 'pomeranian',
 'pug',
 'Ragdoll',
 'Russian_Blue',
 'saint_bernard',
 'samoyed',

 'scottish_terrier',
 'shiba_inu',
 'Siamese',
 'Sphynx',
 'staffordshire_bull_terrier',
 'wheaten_terrier'
 'yorkshire_terrier'
]

	
while True:
	ret, frame = cap.read()
	if not ret: continue 

	height, width, _ = frame.shape


	ch = cv2.waitKey(1) & 0xFF
	if ch == 27: break


	
	n, c, h, w = net.inputs[input_blob].shape

	images = np.ndarray(shape=(n, c, h, w))
	images[0] = cv2.resize(frame, (300,  300)).transpose((2,0,1))


	res = exec_net.infer(inputs={input_blob: images})
	detections = res[out_blob][0][0]

	for i, detect in enumerate(detections):   # detections: (100, 7)  img_id, label_index, confidence

		image_id = float(detect[0])
		label_index = int(detect[1])
		confidence = float(detect[2])


		if image_id < 0 or confidence == 0.:
			continue 


		if confidence > 0.9:
			print("detect: {}, {}".format(labels[label_index-1], confidence)) 
			green = (0, 255, 0)  #BGR
			x_min = int(width*detect[3])
			y_min = int(height*detect[4])
			x_max = int(width*detect[5])
			y_max = int(height*detect[6])
			cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), green, 4)
			cv2.rectangle(frame, (x_min-5, y_min-20), (x_min+105, y_min), green, cv2.FILLED)
			cv2.putText(frame, "%s: %.3f"%(labels[label_index-1], confidence), (x_min, y_min-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), thickness=2, lineType=cv2.LINE_AA)

	cv2.imshow('view', frame)
