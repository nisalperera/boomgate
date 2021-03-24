import numpy as np
import cv2
from detections import license_plates_detection


vehicle_net = cv2.dnn.readNet("./data_utils/weight_files/yolov4.weights","./data_utils/cfg_files/yolov4.cfg")

classes = []

with open("./data_utils/classes/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

vehicle_layer_names = vehicle_net.getLayerNames()
vehicle_output_layers = [vehicle_layer_names[i[0] - 1] for i in vehicle_net.getUnconnectedOutLayers()]
allowed_classes = ["car", "bus", "truck", "three_wheel"]
color = (0, 0, 255)
font = cv2.FONT_HERSHEY_PLAIN

bbox_list = []
threshold_line = 200


def moving_side(new, bbox):
    if new:
        bbox_list.clear()
        return ""
    elif len(bbox_list) == 0:
        bbox_list.append(bbox)
        return ""
    else:
        prev_bbox = bbox_list[-1]
        if prev_bbox[1] > bbox[1] and prev_bbox[1]:
            return "incoming"
        elif prev_bbox[0] < bbox[0]:
            return "outgoing"
        bbox_list.append(bbox)


def detection(image):
    height, width, channel = image.shape
    vehicle_blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    vehicle_net.setInput(vehicle_blob)
    vehicles_outs = vehicle_net.forward(vehicle_output_layers)

    class_ids = []
    confidences = []
    boxes = []

    for out in vehicles_outs:
        for detection in out:
            # print(detection)
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.2 and not class_id == 0:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.85, 0.90)

    original_img = vehicle = image
    plate_confidence = []
    ocr_confidence = []
    vehicle_confidence = []

    for i in range(len(boxes)):
        if i in indexes and str(classes[class_ids[i]]) in allowed_classes:
            x1, y1, w1, h1 = boxes[i]
            if y1 + int(h1/2) < threshold_line:
                label = str(classes[class_ids[i]])
                # print(label)
                # vehicle = image[y+int(h/2):y+h, x:x+w]
                if x1 > 0 and y1 > 0 and w1 > 0 and h1 > 0:
                    vehicle = image[y1:y1+h1, x1:x1+w1]
                vehicle_grayed = cv2.cvtColor(vehicle, cv2.COLOR_BGR2GRAY)
                original_img, plate_confidence, ocr = license_plates_detection.detect_license_plate(vehicle_grayed, image, boxes[i])
                side = moving_side(False, boxes[i][:2])
                vehicle_confidence.append((label, confidences[i], side))
                if ocr is not None:
                    ocr_confidence.append(ocr)
                cv2.rectangle(original_img, (x1, y1), (x1 + w1, y1 + h1), color, 1)
                cv2.line(original_img, (0, threshold_line), (width, threshold_line), (255, 0, 0), 1)
                cv2.putText(original_img, label, (x1, y1 - 5), font, 1, color, 1)
                # print("Vehicle Y coodinate: ", y1 + (h1/2))
            else:
                pass

    return original_img, vehicle_confidence, ocr_confidence, plate_confidence
