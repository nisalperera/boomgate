from database.model import VehicleDetails
from read_video import read_video

vehicleDetails = VehicleDetails()

if __name__ == '__main__':
    read_video()
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
