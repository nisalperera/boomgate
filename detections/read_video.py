import cv2
from detections.vehicle_detection import detection
import time


def select_best(v_confs, p_confs, ocr_confs):
    print()


def read_video():
    # cap = cv2.VideoCapture("./data_utils/video_files/6.mp4")
    cap = cv2.VideoCapture("./data_utils/video_files/cctv_2.mp4")
    print(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ret = True
    while ret:
        start_time = time.time()
        ret, frame = cap.read()
        height, width, channels = frame.shape
        print("Height: {h}, Width: {w}".format(h=height, w=width))
        image = cv2.resize(frame, None, fx=0.4, fy=0.4)[int(height*0.093):, int(width*0.028):int(width*0.140)]
        # image = cv2.resize(frame, None, fx=0.4, fy=0.4)[200:, 110:540]
        # image = cv2.resize(frame, None, fx=0.4, fy=0.4)[200:, :]
        # image = cv2.resize(frame, None, fx=0.4, fy=0.4)
        original, vehicle_confidence, ocr_confidence, plate_confidence = detection(image)

        print("Vehicle confidence: ", vehicle_confidence)
        print("Plate confidence: ", plate_confidence)
        print("OCR confidence: ", ocr_confidence)
        cv2.imshow("Image", original)

        print("FPS: ", 1/(time.time() - start_time))
        print("Time to execute: ", time.time() - start_time)
        key = cv2.waitKey(1)
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
