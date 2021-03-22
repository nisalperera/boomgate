import os
from database.dbconnection import MongoDBConnection
from database.model import VehicleDetails
from datetime import datetime

vehicleDetails = VehicleDetails()
vehicleDetails.date_time = datetime.now()
vehicleDetails.vehicle_type = "Car"
vehicleDetails.license_plate = "HQ 2560"
vehicleDetails.direction_in = True
vehicleDetails.vehicle_image_path = os.path.join(os.getcwd(), "images/vehicles")
vehicleDetails.plate_image_path = os.path.join(os.getcwd(), "images/license_plates")

if __name__ == '__main__':
    connectDb = MongoDBConnection()
    client = connectDb.connect("localhost:27017", database="boomgate")
    # try:
    print(client.server_info())
    print("Connection Success")
    print(vehicleDetails.save())
    print(connectDb.getDataObjs(VehicleDetails))
    print(connectDb.disconnect("boomgate"))
