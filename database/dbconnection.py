from mongoengine import connect, disconnect
import logging


class MongoDBConnection:
    def __init__(self, host=None, database=None, username=None, password=None):
        self.db_host = host
        self.database = database
        self.username = username
        self.password = password

    def connect(self, host=None, database=None, username=None, password=None):
        if self.db_host is None:
            if host is not None:
                self.db_host = host
            else:
                logging.error("Database Url must be given either when initialising Object or when calling this method")
                raise ValueError("Database Url must be given either when initialising Object or when calling this method")

        if self.database is None:
            if database is not None:
                self.database = database
            else:
                logging.error("Database name must be given either when initialising Object or when calling this method")
                raise ValueError("Database name must be given either when initialising Object or when calling this method")

        if self.username is None and self.password is None:
            if username is not None and password is not None:
                self.username = username
                self.password = password
                return connect("mongodb://{username}:{password}@{host}/{database}?authSource=admin".format(
                    username=self.username,
                    password=self.password,
                    host=self.db_host,
                    database=self.database
                ))
            else:
                logging.warning(msg="Database username and password are not given. Using without authentication")
                hostname = self.db_host.split(":")[0]
                port = self.db_host.split(":")[1]
                return connect(self.database, host=hostname, port=int(port), alias=self.database)
        else:
            return connect("mongodb://{username}:{password}@{host}/{database}?authSource=admin".format(
                username=self.username,
                password=self.password,
                host=self.db_host,
                database=self.database
            ), alias=self.database)

    def saveToDb(self, modelObject):
        return modelObject.save()

    def getDataObjs(self, modelObject):
        return modelObject.objects()

    def disconnect(self, db):
        return disconnect(alias=db)

    def getAllDbs(self, client):
        return client.list_collection_names()

