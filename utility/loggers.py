import logging 
import os 
from datetime import datetime
from utility.exception import CustomException
import sys

#Directory creation
directory ="logs_folder"
if not os.path.exists(directory):
    os.mkdir("logs_folder")

#file creation
now = datetime.now()
formatted = now.strftime("%Y-%m-%d_%H-%M-%S")
file_name = os.path.join(directory, f"{formatted}.log")

logging.basicConfig(filename=file_name,format='%(asctime)s-  %(levelname)s- %(message)s',filemode="w")

logger = logging.getLogger()
logger.setLevel(logging.INFO)

if __name__ == "__main__":
    try:
        logger.info("starting")
        logger.info("finished")
        5/0
    except Exception as e:
        logger.error("Error occured")
        print(CustomException(e,sys))


