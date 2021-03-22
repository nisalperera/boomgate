from mongoengine import *
from datetime import datetime


class VehicleDetails(Document):
    date_time = DateTimeField(default=datetime.now())
    vehicle_type = StringField(required=True)
    license_plate = StringField(required=True)
    direction_in = BooleanField(required=True)
    vehicle_image_path = StringField(required=True)
    plate_image_path = StringField(required=True)

    meta = {'db_alias': 'boomgate'}
