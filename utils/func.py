import os
import cv2
import sys
import warnings
import numpy as np
import pandas as pd
import tensorflow as tf
import six.moves.urllib as urllib
from pycocotools.coco import COCO
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

warnings.filterwarnings('ignore', category=FutureWarning)


# patch tf1 into `utils.ops`
utils_ops.tf = tf.compat.v1

# Patch the location of gfile
tf.gfile = tf.io.gfile

def get_pretrained_model(model_name):
    BASE_URL = 'http://download.tensorflow.org/models/object_detection/'
    MODEL_FILE = model_name + '.tar.gz'                       
                                                           
    DOWNLOAD_URL = BASE_URL + MODEL_FILE
    DOWNLOAD_PATH = './pretrained_model'

    if not os.path.isdir(DOWNLOAD_PATH):
        os.mkdir(DOWNLOAD_PATH)
    
    opener = urllib.request.URLopener()
    opener.addheader('User-Agent', 'ballentain')
    opener.retrieve(DOWNLOAD_URL, filename = DOWNLOAD_PATH + '/' + MODEL_FILE)                      
    print('[SUCCESS] Download Complete...') 

def run_inference_for_single_image(image, graph):
    with tf.Session(graph=graph) as sess:

        input_tensor = graph.get_tensor_by_name('image_tensor:0')

        target_operation_names = ['num_detections', 'detection_boxes',
                                  'detection_scores', 'detection_classes', 'detection_masks']
        tensor_dict = {}
        for key in target_operation_names:
            op = None
            try:
                op = graph.get_operation_by_name(key)

            except:
                continue

            tensor = graph.get_tensor_by_name(op.outputs[0].name)
            tensor_dict[key] = tensor

        if 'detection_masks' in tensor_dict:
            detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])

        output_dict = sess.run(tensor_dict, feed_dict={input_tensor: [image]})

        output_dict['num_detections'] = int(output_dict['num_detections'][0])
        output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
        output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
        output_dict['detection_scores'] = output_dict['detection_scores'][0]

        return output_dict

def draw_bounding_boxes(img, output_dict, class_info):
    height, width, _ = img.shape
 
    obj_index = output_dict['detection_scores'] > 0.9
    
    scores = output_dict['detection_scores'][obj_index]
    boxes = output_dict['detection_boxes'][obj_index]
    classes = output_dict['detection_classes'][obj_index]
 
    for box, cls, score in zip(boxes, classes, scores):
        # draw bounding box
        img = cv2.rectangle(img,
                            (int(box[1] * width), int(box[0] * height)),
                            (int(box[3] * width), int(box[2] * height)), class_info[cls][1], 8)
 
        # put class name & percentage
        object_info = class_info[cls][0] + ': ' + str(int(score * 100)) + '%'
        text_size, _ = cv2.getTextSize(text = object_info,
                                       fontFace = cv2.FONT_HERSHEY_SIMPLEX,
                                       fontScale = 0.9, thickness = 2)
        img = cv2.rectangle(img,
                            (int(box[1] * width), int(box[0] * height) - 25),
                            (int(box[1] * width) + text_size[0], int(box[0] * height)),
                            class_info[cls][1], -1)
        img = cv2.putText(img,
                          object_info,
                          (int(box[1] * width), int(box[0] * height)),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
        
    return img

def show_detection_details(output_dict):
    print('Number of Detections : {}'.format(output_dict['num_detections']))
    print('Detection Boxes : {}'.format(output_dict['detection_boxes']))
    print('Detection Scores : {}'.format(output_dict['detection_scores']))
    print('Detection Classes : {}'.format(output_dict['detection_classes']))


def get_coco_class_label():
    class_info = {}
    with open('./pretrained_model/coco_class.txt', 'r') as f:
        for line in f:
            info = line.split(', ')
 
            class_index = int(info[0])
            class_name = info[1]
            color = (int(info[2][1:]), int(info[3]), int(info[4].strip()[:-1]))    
    
            class_info[class_index] = [class_name, color]
    return class_info

def get_label_category_index(path_to_label,num_classes):
    label_map = label_map_util.load_labelmap(path_to_label)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=num_classes, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    return label_map, categories, category_index

def get_image_from_coco(category):
    # Initializations (make them as arguments)
    cat_of_interest = category # supports only one category.
    ann_file = "./data/mscoco_dataset/annotations/instances_train2017.json"
    csv_save_path = "./data/{}.csv".format(cat_of_interest)

    # COCO instance
    coco = COCO(ann_file)

    # get category id
    cat_id  = coco.getCatIds(catNms=[cat_of_interest])

    # get annotation ids for current category
    ann_ids = coco.getAnnIds(catIds=cat_id, iscrowd=None)
    all_ann = coco.loadAnns(ann_ids)

    # Loop through each annotation and create a data frame with necessary
    #     information to create csv file. This file later aids in creating
    #     tensorflow record.

    df_rows = []
    for i in range(0, len(all_ann)):
        cur_ann    = all_ann[i]
        cbbox      = cur_ann["bbox"]
        cimg_info  = coco.loadImgs(cur_ann["image_id"])

        if(len(cimg_info) > 1):
            print("ERROR: More than one image got loaded")
            sys.exit(1)

        filename   = cimg_info[0]["file_name"]
        cur_class  = cat_of_interest
        width    = cimg_info[0]["width"]
        height   = cimg_info[0]["height"]
        xmin     = int(cbbox[0])
        ymin     = int(cbbox[1])
        xmax     = min(int(xmin + cbbox[2]), width-1)
        ymax     = min(int(ymin + cbbox[3]), height-1)

        df_rows  = df_rows + [[filename, str(width), str(height), cur_class,
                               str(xmin), str(ymin), str(xmax), str(ymax)]]

    df=pd.DataFrame(df_rows, columns=["filename", "width", "height", "class",
                               "xmin", "ymin", "xmax", "ymax"])
    df.to_csv(csv_save_path, index=None)