import numpy as np
import cv2
from detections import license_plates_detection
from detections.tracking import ObjectTracker
import logging
import torch
import torchvision


logger = logging.getLogger(__name__)
# vehicle_net = cv2.dnn.readNet("./data_utils/weight_files/yolov4.weights","./data_utils/cfg_files/yolov4.cfg")
vehicle_net = cv2.dnn.readNet("./data_utils/weight_files/best.onnx")
#
classes = []

with open("./data_utils/classes/vehicles.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

vehicle_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
vehicle_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)

model = cv2.dnn_DetectionModel(vehicle_net)
model.setInputParams(size=(416, 416), scale=1/255)

# vehicle_layer_names = vehicle_net.getLayerNames()
# vehicle_output_layers = [vehicle_layer_names[i[0] - 1] for i in vehicle_net.getUnconnectedOutLayers()]
allowed_classes = ["car", "bus", "truck", "three_wheel"]
color = (0, 0, 255)
font = cv2.FONT_HERSHEY_PLAIN

bbox_list = []
y_cods = []
# threshold_line = 200

tracker = ObjectTracker(100)


def moving_side(bbox, threshold_line):
    # if new:
    #     y_cods.clear()
    #     bbox_list.clear()
    #     return None
    # elif len(bbox_list) == 0:
    #     y_cods.append(bbox[1])
    #     bbox_list.append(bbox[1])
    #     return None
    # else:
        # prev_bbox = bbox_list[-1]
        # bbox_list.append(bbox[1])
        # y_cods.append(bbox[1])
        # if len(y_cods) >= 2:
        # moving_avg = np.convolve(np.array(y_cods), np.ones(10), 'valid') / 2
        # avg = sum(moving_avg) / len(moving_avg)
        # logger.info("   Moving Average: {avg}".format(avg=round(avg, 2)))
        # logger.info("   Moving Average: {avg}".format(avg=moving_avg))
    if bbox[1] < threshold_line:
        return True
    elif bbox[1] > threshold_line:
        return False


def detection(image, threshold_line=200):
    height, width, channel = image.shape
    # vehicle_blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    # vehicle_net.setInput(vehicle_blob)
    # vehicles_outs = vehicle_net.forward(vehicle_output_layers)

    class_ids, confidences, boxes = model.detect(image, 0.9, 0.5)
    class_ids_filtered, confidences_filtered, boxes_filtered = [], [], []

    for i, class_id in enumerate(class_ids):
        if classes[class_id[0]] in allowed_classes:
            class_ids_filtered.append(class_id[0])
            confidences_filtered.append(confidences[i])
            boxes_filtered.append(boxes[i].tolist())

    # class_i-_car = class_ids.tolist().index(2)

    original_img = vehicle = image
    # plate_confidence = []
    # ocr_confidence = []
    detections = []
    # centroids = []
    # calculate_max = False

    stop_detection = False
    license_plate = None

    objects = tracker.update(np.array(boxes_filtered))
    if objects is not None:
        for (objectID, centroid) in objects.items():
            # draw both the ID of the object and the centroid of the
            # object on the output frame
            bbox = centroid[2:].tolist()
            if isinstance(centroid, np.ndarray) and isinstance(boxes_filtered, list) and any(box == bbox for box in boxes_filtered):
                index = boxes_filtered.index(centroid[2:].tolist())
                x1, y1, w1, h1 = boxes_filtered[index]
                # if classes[class_ids[index][0]] in allowed_classes:
                if centroid[1] < threshold_line:
                    # text = "ID {}".format(objectID)
                    label = str(classes[class_ids_filtered[index]])

                    if x1 > 0 and y1 > 0 and w1 > 0 and h1 > 0:
                        vehicle = image[y1:y1 + h1, x1:x1 + w1]
                    vehicle_grayed = cv2.cvtColor(vehicle, cv2.COLOR_BGR2GRAY)
                    original_img, license_plate, plate_confidence, ocr = license_plates_detection.detect_license_plate(vehicle_grayed,
                                                                                                                       vehicle,
                                                                                                                       image,
                                                                                                                       boxes_filtered[index])
                    side = moving_side(bbox, threshold_line)
                    if ocr is None:
                        # ocr_confidence.append(ocr)
                        ocr = []
                    else:
                        pass

                    # vehicle_confidence.append([(label, confidences[i], side), plate_confidence, ocr])
                    detections += [[label, confidences_filtered[index][0], side], plate_confidence, ocr]

                    cv2.rectangle(original_img, (x1, y1), (x1 + w1, y1 + h1), color, 1)
                    # cv2.line(original_img, (0, threshold_line), (width, threshold_line), (255, 0, 0), 1)
                    cv2.putText(original_img, label, (x1, y1 - 5), font, 1, color, 1)
                    # cv2.putText(original_img, text, (centroid[0] - 10, centroid[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    cv2.circle(original_img, (centroid[0], centroid[1]), 2, (0, 255, 0), -1)
                else:
                    # if centroid[1] > threshold_line:
                    # stop_detection = True
                    # side = moving_side(False, bbox)
                    # if not side and centroid[1] > threshold_line:
                    #     stop_detection = False
                    # elif not side and centroid[1] < threshold_line:
                    #     stop_detection = True
                    # elif side and centroid[1] > threshold_line:
                    #     stop_detection = True
                    # elif side and centroid[1] < threshold_line:
                    #     stop_detection = False
                    # break
                    side = moving_side(bbox, threshold_line)
                    if not side and centroid[1] > threshold_line:
                        stop_detection = True
                    elif not side and centroid[1] < threshold_line:
                        stop_detection = False
                    elif side and centroid[1] > threshold_line:
                        stop_detection = True
                    else:
                        stop_detection = False
                    break
                # logger.info("Direction: {dir}".format(dir=side))

    return original_img, vehicle, license_plate, detections, stop_detection
    # return original_img
