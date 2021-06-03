import numpy as np
from collections import OrderedDict
from scipy.spatial import distance as dist
import cv2


# # OPENCV_OBJECT_TRACKERS = {
# #     "kcf": cv2.Tracker,
# #     "boosting": cv2.TrackerBoosting_create,
# #     "mil": cv2.TrackerMIL_create,
# #     "tld": cv2.TrackerTLD_create,
# #     "medianflow": cv2.TrackerMedianFlow_create,
# #     "mosse": cv2.TrackerMOSSE_create
# # }
#
#
class ObjectTracker:
    def __init__(self, distance):
        self.distance = distance
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.nextObjectID = 0

    def register(self, centroid):
        # when registering an object we use the next available object
        # ID to store the centroid
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        # to deregister an object ID we delete the object ID from
        # both of our respective dictionaries
        del self.objects[objectID]
        del self.disappeared[objectID]

    def update(self, rects):
        # check to see if the list of input bounding box rectangles
        # is empty
        try:
            if len(rects) == 0:
                # loop over any existing tracked objects and mark them
                # as disappeared
                for objectID in list(self.disappeared.keys()):
                    self.disappeared[objectID] += 1
                    # if we have reached a maximum number of consecutive
                    # frames where a given object has been marked as
                    # missing, deregister it
                    if self.disappeared[objectID] > self.distance:
                        self.deregister(objectID)
                # return early as there are no centroids or tracking info
                # to update
                return self.objects

            # print("Rectangles Shape: ", rects.shape)
            inputCentroids = np.zeros((len(rects), 2 + rects.shape[1]), dtype="int")
            # loop over the bounding box rectangles
            for (i, (startX, startY, width, height)) in enumerate(rects):
                # use the bounding box coordinates to derive the centroid
                cX = int(((2 * startX) + width) / 2.0)
                cY = int(((2 * startY) + height) / 2.0)
                inputCentroids[i] = np.array([cX, cY] + rects[i].tolist())

            if len(self.objects) == 0:
                for i in range(0, len(inputCentroids)):
                    self.register(inputCentroids[i])

            else:
                objectIDs = list(self.objects.keys())
                objectCentroids = np.array(list(self.objects.values()))[:, :2]

                # print("Object Centroids: ", objectCentroids)
                # compute the distance between each pair of object
                # centroids and input centroids, respectively -- our
                # goal will be to match an input centroid to an existing
                # object centroid
                D = dist.cdist(objectCentroids, inputCentroids[:, :2])
                # in order to perform this matching we must (1) find the
                # smallest value in each row and then (2) sort the row
                # indexes based on their minimum values so that the row
                # with the smallest value is at the *front* of the index
                # list
                rows = D.min(axis=1).argsort()
                # next, we perform a similar process on the columns by
                # finding the smallest value in each column and then
                # sorting using the previously computed row index list
                cols = D.argmin(axis=1)[rows]

                # in order to determine if we need to update, register,
                # or deregister an object we need to keep track of which
                # of the rows and column indexes we have already examined
                usedRows = set()
                usedCols = set()
                # loop over the combination of the (row, column) index
                # tuples
                for (row, col) in zip(rows, cols):
                    # if we have already examined either the row or
                    # column value before, ignore it
                    # val
                    if row in usedRows or col in usedCols:
                        continue
                    # otherwise, grab the object ID for the current row,
                    # set its new centroid, and reset the disappeared
                    # counter
                    objectID = objectIDs[row]
                    self.objects[objectID] = inputCentroids[col]
                    self.disappeared[objectID] = 0
                    # indicate that we have examined each of the row and
                    # column indexes, respectively
                    usedRows.add(row)
                    usedCols.add(col)

                    # compute both the row and column index we have NOT yet
                    # examined
                    unusedRows = set(range(0, D.shape[0])).difference(usedRows)
                    unusedCols = set(range(0, D.shape[1])).difference(usedCols)

                    if D.shape[0] >= D.shape[1]:
                        # loop over the unused row indexes
                        for row in unusedRows:
                            # grab the object ID for the corresponding row
                            # index and increment the disappeared counter
                            objectID = objectIDs[row]
                            self.disappeared[objectID] += 1
                            # check to see if the number of consecutive
                            # frames the object has been marked "disappeared"
                            # for warrants deregistering the object
                            if self.disappeared[objectID] > self.distance:
                                self.deregister(objectID)

                    else:
                        for col in unusedCols:
                            self.register(inputCentroids[col])
                        # return the set of trackable objects
                return self.objects

        except Exception as ex:
            print(ex)
            pass

# (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')￼

# def tracker():
#
#     # Set up tracker.
#     # Instead of MIL, you can also use
#
#     (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
#
#     tracker_types = ['KCF', 'CSRT']
#     tracker_type = tracker_types[0]
#
#     if int(major_ver) < 4 and int(minor_ver) < 3:
#         tracker = cv2.cv2.Tracker_create(tracker_type)
#     else:
#         if tracker_type == 'KCF':
#             tracker = cv2.TrackerKCF_create()
#         if tracker_type == 'CSRT':
#             tracker = cv2.TrackerCSRT_create()
#
#     # Read video
#     video = cv2.VideoCapture("./data_utils/video_files/cctv_2.mp4")
#
#     # Exit if video not opened.
#     if not video.isOpened():
#         print("Could not open video")
#
#     # Read first frame.
#     ok, frame = video.read()
#     if not ok:
#         print('Cannot read video file')
#
#     # Define an initial bounding box
#     # bbox = (287, 23, 86, 320)
#
#     frame = cv2.resize(frame, None, fx=0.4, fy=0.4)
#
#     # Uncomment the line below to select a different bounding box
#     bbox = cv2.selectROI(frame, False)
#
#     # Initialize tracker with first frame and bounding box
#     ok = tracker.init(frame, bbox)
#
#     while True:
#         # Read a new frame
#         ok, frame = video.read()
#
#         frame = cv2.resize(frame, None, fx=0.4, fy=0.4)
#
#         if not ok:
#             break
#
#         # Start timer
#         timer = cv2.getTickCount()
#
#         # Update tracker
#         ok, bbox = tracker.update(frame)
#
#         # Calculate Frames per second (FPS)
#         fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
#
#         # Draw bounding box
#         if ok:
#             # Tracking success
#             p1 = (int(bbox[0]), int(bbox[1]))
#             p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
#             cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
#         else:
#             # Tracking failure
#             cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
#
#         # Display tracker type on frame
#         cv2.putText(frame, tracker_type + " Tracker", (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2);
#
#         # Display FPS on frame
#         cv2.putText(frame, "FPS : " + str(int(fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2);
#
#         # Display result
#         cv2.imshow("Tracking", frame)
#
#         # Exit if ESC pressed
#         k = cv2.waitKey(1) & 0xff
#         if k == 27: break
