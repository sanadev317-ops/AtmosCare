# backend/realtime_listener.py
import threading
from pymongo import MongoClient
from Backend.config import DATABASE_URI, DATABASE_NAME

def listen_for_changes(callback):
    client = MongoClient(DATABASE_URI)
    db = client[DATABASE_NAME]
    collection = db["messages"]

    # Start listening for changes
    with collection.watch() as stream:
        for change in stream:
            callback(change["fullDocument"])

def start_listener(callback):
    thread = threading.Thread(target=listen_for_changes, args=(callback,), daemon=True)
    thread.start()
