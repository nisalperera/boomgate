import cv2
import numpy as np
import easyocr

reader = easyocr.Reader(['en'])

with open("./data_utils/classes/license_plate.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

license_plate_net = cv2.dnn.readNet("./data_utils/weight_files/yolov4-tiny-detector-1ch_final.weights",
                                    "./data_utils/cfg_files/yolov4-tiny-detector-1ch.cfg")

license_plate_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
license_plate_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)

# license_layer_names = license_plate_net.getLayerNames()
# license_output_layers = [license_layer_names[i[0] - 1] for i in license_plate_net.getUnconnectedOutLayers()]

model = cv2.dnn_DetectionModel(license_plate_net)
model.setInputParams(size=(416, 416), scale=1/255)

pad = 20
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
    # license_blob = cv2.dnn.blobFromImage(image, 1/255, (416, 416), (0, 0), True, crop=False)
    # license_plate_net.setInput(license_blob)
    # licenses = license_plate_net.forward(license_output_layers)

    class_ids, confidences, boxes = model.detect(image, 0.8, 0.5)

    license_plate_im = None
    plate_only_resu = None
    plate_ocr = None
    plate_confidence = []
    best_plate = 0
    plate_max_conf = 0

    for i in range(len(boxes)):
        # if i in indexes:
            # conf = confidences[class_ids[i]]
            # print(conf)
        x2, y2, w2, h2 = boxes[i]
        label = str(classes[class_ids[i][0]]) + ": " + str(confidences[class_ids[i][0]])
        plate_confidence += [classes[class_ids[i][0]], confidences[class_ids[i][0]][0].tolist()]
        # print(label)
        if y2 >= pad and x2 - pad >= 0 and x2 + w2 + pad < width and y2 + h2 + pad < height:
            license_plate_im = image[y2-pad:y2+h2+pad, x2-pad:x2 + w2+pad]

        if license_plate_im is not None:
            license_plate_resized = cv2.resize(license_plate_im, (license_plate_im.shape[1] * scale, license_plate_im.shape[0] * scale),
                                               interpolation=cv2.INTER_LINEAR)
            plate_ocr = reader.readtext(license_plate_resized)
            plate_ocr = ocr_with_max_conf(plate_ocr)
            plate_only_resu = [*plate_ocr]
            # print("OCR Out License Plate Image: ", plate_only_resu)
        else:
            plate_ocr = reader.readtext(image)
            plate_ocr = ocr_with_max_conf(plate_ocr)
            plate_only_resu = [*plate_ocr]
            x, y = x2 + vehicle_bbox[0], y2 + vehicle_bbox[1]
            cv2.rectangle(original_im, (x, y), (x + w2, y + h2), color, 1)
            cv2.putText(original_im, label, (x, y - 5), font, 1, color, 1)

    # cv2.imshow("Image", license_plate_resized)
    if plate_ocr is not None and len(plate_ocr[0]) >= 6:
        license_plate = plate_ocr[0].split(" ")
        if len(license_plate) == 1:
            if 7 >= len(license_plate) >= 6:
                plate_only_resu[0] = license_plate
            else:
                # plate_only_resu[0] = license_plate
                plate_only_resu[0] = license_plate[2:]
        elif len(license_plate) == 2:
            if 5 > len(license_plate[0]) > 3:
                plate_only_resu[0] = license_plate
            else:
                plate_only_resu[0] = license_plate[0][2:] + license_plate[1]

        return original_im, license_plate_im, plate_confidence, plate_only_resu
    else:
        return original_im, license_plate_im, plate_confidence, []
