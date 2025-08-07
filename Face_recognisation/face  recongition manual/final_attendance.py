import cv2
# import numpy as np
import csv
from datetime import datetime

#current date and time
now =datetime.now()
current_date=now.strftime("%Y-%m-%d")

#open csv file 
f=open(f"{current_date}.csv", "w+", newline="")
lnwritter=csv.writer(f)

def give_attendance(names):
    print(names)
    print(current_date)