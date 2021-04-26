import os
from database.dbconnection import MongoDBConnection
from database.model import VehicleDetails
from datetime import datetime
import easyocr


if __name__ == '__main__':
    
    print(easyocr.__version__)