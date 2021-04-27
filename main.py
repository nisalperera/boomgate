from database.model import VehicleDetails
from detections.read_video import read_video, read_image
# from detections.tracking import tracker

vehicleDetails = VehicleDetails()
image_path = "./data_utils/images/frame137.jpg"

if __name__ == '__main__':
    read_video()
    # read_image("./data_utils/images/frame137.jpg")
    # tracker()
    # connectDb = MongoDBConnection()
    # client = connectDb.connect("localhost:27017", database="boomgate")
    # try:
    #     print(client.server_info())
    #     print("Connection Success")
    #     print(vehicleDetails.save())
    #     print(connectDb.getDataObjs(VehicleDetails))
    #     print(connectDb.disconnect("boomgate"))
    # except Exception as con_ex:
    #     print("Connection Failed")
    #     print(str(con_ex))
