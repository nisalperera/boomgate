import cv2
import logging
import time
import numpy as np
from datetime import datetime
from detections.vehicle_detection import detection
from database.dbconnection import MongoDBConnection
from database.model import VehicleDetails


loggingFormat = "%(asctime) %(filename) %(message)s"
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
skip_frames = 0

con = MongoDBConnection("localhost:27017", "boomgate")
mongoclient = con.connect()
# vehicledetails = None

vehicle_image_path = "./images/vehicles/{name}"
license_image_path = "./images/license_plates/{name}"

# print(mongoclient.server_info())
confidence_list = []
frame_list = []
license_plate_list = [None]
prev_license_plate = None


def read_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, None, fx=1, fy=1)
    vehicledetails = perform_detection(image, False, None, int(image.shape[0]))
    # cv2.imshow("Image", original)
    # logger.info("   Detection: {detections}".format(detections=vehicle_confidence))

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def getAllObjs(vehicledetails):
    return con.getDataObjs(vehicledetails)


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
    return vehicledetails


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


def perform_detection(image, save, start_time, threshold_line=200):
    height, width, channels = image.shape
    vehicledetails = None
    original, vehicle, license_plate, vehicle_confidence, stop_detection = detection(image, threshold_line)
    confidence_list.append(vehicle_confidence)

    frame_list.append(vehicle)
    if license_plate is None:
        if len(license_plate_list) > 0 and license_plate_list[-1] is not None:
            license_plate_list.append(license_plate_list[-1])
        else:
            license_plate_list.append(None)
    else:
        license_plate_list.append(license_plate)
    if stop_detection:
        if save:
            logger.info("   Selecting best predictions.")
            max_confidences, index = select_best(confidence_list)
            if len(max_confidences) > 0 and len(max_confidences[0]) > 0:
                logger.info("   Max Vehicle confidence: {conf}".format(conf=max_confidences))
                logger.info("   Saving data to Database")
                date_time = datetime.now()
                # file_name = max_confidences[0][0] + str(date_time) + ".jpg"
                file_name = max_confidences[0][0] + date_time.strftime("%Y%m%d%H%M%S") + ".jpg"
                vehicledetails = saveToDB(max_confidences, date_time, file_name)
                cv2.imwrite(vehicle_image_path.format(name=file_name), frame_list[index])
                if license_plate_list[index] is not None:
                    cv2.imwrite(license_image_path.format(name=file_name), license_plate_list[index])
                else:
                    cv2.imwrite(license_image_path.format(name=file_name), np.zeros(frame_list[index].shape))
                save = False
                frame_list.clear()
                confidence_list.clear()
                license_plate_list.clear()

                if skip_frames == 0:
                    logger.info("   FPS: {fps}".format(fps=1 / (time.time() - start_time)))
                else:
                    logger.info("   FPS: {fps}".format(fps=skip_frames / (time.time() - start_time)))
                logger.info("   Time to execute: {exect}".format(exect=time.time() - start_time))
    else:
        save = True
        # cv2.imshow("Image", original)
        # print("Vehicle confidence: ", confidence_list)
        # print("Plate confidence: ", plate_confidence)
        # print("OCR confidence: ", ocr_confidence)

    # if license_plate is not None and license_plate.shape[0] > 0 and license_plate.shape[1] > 0:
    #     cv2.imshow("Image", original)
    # else:
    cv2.line(original, (0, threshold_line), (original.shape[1], threshold_line), (255, 0, 0), 1)
    cv2.imshow("Image", original)

    return vehicledetails


def read_video():
    # cap = cv2.VideoCapture("./data_utils/video_files/cctv_2.mp4")
    cap = cv2.VideoCapture("./data_utils/video_files/6.mp4")
    # print(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count = 0
    save = True
    vehicledetails = None

    while True:
        start_time = time.time()
        ret, frame = cap.read()

        if not ret:
            break

        # print("Height: {h}, Width: {w}".format(h=height, w=width))
        frame_count += 1
        image = cv2.resize(frame, None, fx=0.4, fy=0.4)
        height, width, channels = image.shape
        # image = cv2.resize(frame, None, fx=0.4, fy=0.4)[int(height*0.093):, int(width*0.028):int(width*0.140)]

        if skip_frames > 0:
            if frame_count % skip_frames == 0:
                # image = cv2.resize(frame, None, fx=0.4, fy=0.4)[200:, 110:540]
                # image = cv2.resize(frame, None, fx=0.4, fy=0.4)[200:, :]
                # image = cv2.resize(frame, None, fx=0.4, fy=0.4)
                vehicledetails = perform_detection(image, save, start_time, int(height * 0.463))

        else:
            vehicledetails = perform_detection(image, save, start_time, int(height * 0.463))
            # image = cv2.resize(frame, None, fx=0.4, fy=0.4)[200:, 110:540]
            # image = cv2.resize(frame, None, fx=0.4, fy=0.4)[200:, :]
            # image = cv2.resize(frame, None, fx=0.4, fy=0.4)
            # original, license_plate, vehicle_confidence = detection(image)
            #
            # print("Vehicle confidence: ", vehicle_confidence)
            # # print("Plate confidence: ", plate_confidence)
            # # print("OCR confidence: ", ocr_confidence)
            # if license_plate is not None and license_plate.shape[0] > 0 and license_plate.shape[1] > 0:
            #     cv2.imshow("Image", original)
            # else:
            #     cv2.imshow("Image", original)
            #
            # print("FPS: ", 1 / (time.time() - start_time))
            # print("Time to execute: ", time.time() - start_time)
        key = cv2.waitKey(1)
        if key == 27:
            break

    try:
        logger.info("   {objs}".format(objs=getAllObjs(vehicledetails)))
    except Exception as ex:
        logger.error("  {err}".format(err=str(ex)))

    cap.release()
    cv2.destroyAllWindows()
