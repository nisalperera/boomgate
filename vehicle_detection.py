from datetime import datetime
import numpy as np
import cv2
import license_plates_detection


vehicle_net = cv2.dnn.readNet("yolov4.weights","yolov4.cfg")

classes = []

with open("coco.names","r") as f:
    classes = [line.strip() for line in f.readlines()]

vehicle_layer_names = vehicle_net.getLayerNames()
vehicle_output_layers = [vehicle_layer_names[i[0] - 1] for i in vehicle_net.getUnconnectedOutLayers()]

color = (0, 0, 255)


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

    vehicle = image
    plate = None

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            print(label)
            vehicle = image[y+int(h/2):y+h, x:x+w]
            vehicle_grayed = cv2.cvtColor(vehicle, cv2.COLOR_BGR2GRAY)
            vehicle, plate = license_plates_detection.detect_license_plate(vehicle_grayed, image, color)

    return vehicle, plate
