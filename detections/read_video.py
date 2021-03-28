import cv2
import logging
import time
import numpy as np
from datetime import datetime
from detections.vehicle_detection import detection
from database.dbconnection import MongoDBConnection
from database.model import VehicleDetails


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
skip_frames = 5

con = MongoDBConnection("localhost:27017", "boomgate")
mongoclient = con.connect()

vehicle_image_path = "./images/vehicles/{name}"
license_image_path = "./images/license_plates/{name}"

# print(mongoclient.server_info())


def read_image():
    image = cv2.imread("./data_utils/images/frame137.jpg")
    image = cv2.resize(image, None, fx=0.4, fy=0.4)
    original, vehicle_confidence = detection(image)
    cv2.imshow("Image", original)
    print("Detection: ", vehicle_confidence)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def saveToDB(data, date_time, file_name):
    vehicledetails = VehicleDetails()
    vehicledetails.date_time = date_time
    vehicledetails.vehicle_type = data[0][0]
    # logger.error("Error: {}".format(data))
    if data[2][0] is None or len(data[2][0]) == 0:
        vehicledetails.license_plate = ""
    elif len(data[2][0]) > 1:
        string = ""
        for s in data[2][0]:
            string += s
        vehicledetails.license_plate = string
    else:
        vehicledetails.license_plate = data[2][0]
    vehicledetails.incoming = data[0][2]
    vehicledetails.vehicle_image_path = "./images/vehicles/{name}".format(name=file_name)
    vehicledetails.plate_image_path = "./images/license_plates/{name}".format(name=file_name)
    con.saveToDb(vehicledetails)


def select_best(confidences):
    max_confidence_index = 0
    for i, confidence_list in enumerate(confidences):
        max_confidence = 0
        if len(confidence_list) > 0 and len(confidence_list[2]) > 0 and confidence_list[2][1] > max_confidence and len(confidence_list[2][0]) > 0:
            max_confidence = confidence_list[2][1]
            max_confidence_index = i
        else:
            continue

    return confidences[max_confidence_index], max_confidence_index
    # if len(confidences[:, 2]) > 0:
    #     licence_plate_confidences += confidences[:, 2, 1]


def read_video():
    # cap = cv2.VideoCapture("./data_utils/video_files/6.mp4")
    cap = cv2.VideoCapture("./data_utils/video_files/6.mp4")
    # print(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count = 0

    confidence_list = []
    frame_list = []
    license_plate_list = [None]
    prev_license_plate = None
    save = True

    while True:
        start_time = time.time()
        ret, frame = cap.read()

        if not ret:
            break

        height, width, channels = frame.shape
        # print("Height: {h}, Width: {w}".format(h=height, w=width))
        frame_count += 1
        image = cv2.resize(frame, None, fx=0.4, fy=0.4)
        # image = cv2.resize(frame, None, fx=0.4, fy=0.4)[int(height*0.093):, int(width*0.028):int(width*0.140)]

        if skip_frames > 0:
            if frame_count % skip_frames == 0:
                # image = cv2.resize(frame, None, fx=0.4, fy=0.4)[200:, 110:540]
                # image = cv2.resize(frame, None, fx=0.4, fy=0.4)[200:, :]
                # image = cv2.resize(frame, None, fx=0.4, fy=0.4)
                original, license_plate, vehicle_confidence, stop_detection = detection(image)
                confidence_list.append(vehicle_confidence)

                frame_list.append(original)
                if license_plate is None:
                    if license_plate_list[-1] is not None:
                        license_plate_list.append(license_plate_list[-1])
                    else:
                        license_plate_list.append(None)
                else:
                    license_plate_list.append(license_plate)
                if stop_detection:
                    logger.info("Selecting best predictions.")
                    max_confidences, index = select_best(confidence_list)
                    if save and len(max_confidences) > 0 and len(max_confidences[0]) > 0:
                        logger.info("Max Vehicle confidence: {conf}".format(conf=max_confidences))
                        logger.info("Saving data to Database")
                        date_time = datetime.now()
                        file_name = max_confidences[0][0] + str(date_time) + ".jpg"
                        saveToDB(max_confidences, date_time, file_name)
                        cv2.imwrite(vehicle_image_path.format(name=file_name), frame_list[index])
                        if license_plate_list[index] is not None:
                            cv2.imwrite(license_image_path.format(name=file_name), license_plate_list[index])
                        else:
                            cv2.imwrite(license_image_path.format(name=file_name), np.zeros(frame_list[index].shape))
                        save = False
                else:
                    # print("Vehicle confidence: ", confidence_list)
                    save = True
                    # print("Plate confidence: ", plate_confidence)
                    # print("OCR confidence: ", ocr_confidence)

                # if license_plate is not None and license_plate.shape[0] > 0 and license_plate.shape[1] > 0:
                #     cv2.imshow("Image", original)
                # else:
                cv2.imshow("Image", original)

                logger.info("FPS: {fps}".format(fps=skip_frames/(time.time() - start_time)))
                logger.info("Time to execute: {exect}".format(exect=time.time() - start_time))
        else:
            # image = cv2.resize(frame, None, fx=0.4, fy=0.4)[200:, 110:540]
            # image = cv2.resize(frame, None, fx=0.4, fy=0.4)[200:, :]
            # image = cv2.resize(frame, None, fx=0.4, fy=0.4)
            original, license_plate, vehicle_confidence = detection(image)

            print("Vehicle confidence: ", vehicle_confidence)
            # print("Plate confidence: ", plate_confidence)
            # print("OCR confidence: ", ocr_confidence)
            if license_plate is not None and license_plate.shape[0] > 0 and license_plate.shape[1] > 0:
                cv2.imshow("Image", original)
            else:
                cv2.imshow("Image", original)

            print("FPS: ", 1 / (time.time() - start_time))
            print("Time to execute: ", time.time() - start_time)
        key = cv2.waitKey(1)
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
