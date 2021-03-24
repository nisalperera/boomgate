import cv2
import numpy as np
import easyocr

reader = easyocr.Reader(['en'])

with open("./data_utils/classes/license_plate.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

license_plate_net = cv2.dnn.readNet("./data_utils/weight_files/yolov4-tiny-detector-1ch_final.weights",
                                    "./data_utils/cfg_files/yolov4-tiny-detector-1ch.cfg")

license_layer_names = license_plate_net.getLayerNames()
license_output_layers = [license_layer_names[i[0] - 1] for i in license_plate_net.getUnconnectedOutLayers()]

pad = 50
scale = 1
color = (0, 255, 0)
font = cv2.FONT_HERSHEY_PLAIN


def ocr_with_max_conf(ocr_list):
    if len(ocr_list) > 0:
        max_conf = 0
        max_con_index = 0
        for index, ocr in enumerate(ocr_list):
            if ocr[2] > max_conf:
                max_conf = ocr[2]
                max_con_index = index
        return ocr_list[max_con_index][1:]
    else:
        return []


def detect_license_plate(image, original_im, vehicle_bbox):
    height, width = image.shape
    # resized_license = cv2.resize(image, (416, 416), interpolation=cv2.INTER_LINEAR)
    license_blob = cv2.dnn.blobFromImage(image, 1/255, (416, 416), (0, 0), True, crop=False)
    license_plate_net.setInput(license_blob)
    licenses = license_plate_net.forward(license_output_layers)

    class_ids = []
    confidences = []
    boxes = []

    for output in licenses:
        for license in output:
            confidence_score = license[5:]
            class_id = np.argmax(confidence_score)
            confidence = confidence_score[class_id]

            if confidence > 0.80 and class_id == 0:
                # Object detected
                center_x = int(license[0] * width)
                center_y = int(license[1] * height)
                w = int(license[2] * width)
                h = int(license[3] * height)
                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.90, 0.95)

    license_plate = image
    plate_only_resu = None
    plate_confidence = []
    best_plate = 0
    plate_max_conf = 0

    for i in range(len(boxes)):
        if i in indexes:
            # conf = confidences[class_ids[i]]
            # print(conf)
            x2, y2, w2, h2 = boxes[i]
            label = str(classes[class_ids[i]]) + ": " + str(confidences[class_ids[i]])
            plate_confidence.append(((classes[class_ids[i]]), confidences[class_ids[i]]))
            # print(label)
            if y2 >= pad and x2 - pad >= 0 and x2 + w2 + pad < width and y2 + h2 + pad < height:
                license_plate = image[y2-pad:y2+h2+pad, x2-pad:x2 + w2+pad]

            license_plate_resized = cv2.resize(license_plate, (license_plate.shape[1] * scale, license_plate.shape[0] * scale),
                                               interpolation=cv2.INTER_LINEAR)

            plate_only_resu = reader.readtext(license_plate_resized)
            plate_only_resu = ocr_with_max_conf(plate_only_resu)
            # print("OCR Out License Plate Image: ", plate_only_resu)
            x, y = x2 + vehicle_bbox[0], y2 + vehicle_bbox[1]
            cv2.rectangle(original_im, (x, y), (x + w2, y + h2), color, 1)
            cv2.putText(original_im, label, (x, y - 5), font, 1, color, 1)

    return original_im, plate_confidence, plate_only_resu
