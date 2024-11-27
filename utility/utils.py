import sys
import pickle
from utility.exception import CustomException

def save_object(filename,obj):
    try:
        with open(filename, 'wb') as f:
            pickle.dump(obj, f)
    except Exception as e:
        print(CustomException(e,sys))

def load_object(filename):
    try:
        with open(filename, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(CustomException(e,sys))
