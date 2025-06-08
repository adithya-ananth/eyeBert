import firebase_admin
import json
from firebase_admin import credentials, db

cred_obj = credentials.Certificate("naanbert-firebase-adminsdk-k1t8m-65d91682f0.json")

default_app = firebase_admin.initialize_app(cred_obj, {
    'databaseURL': 'https://naanbert-default-rtdb.firebaseio.com/'
})

ref = db.reference('/')

with open("data.json", "r") as f:
    file_contents = json.load(f)

ref.set(file_contents)

print("Data uploaded successfully!")



