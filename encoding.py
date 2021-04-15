import numpy as np
import pandas as pd
import face_recognition
import cv2
import os

path = 'Images'
images = []
className = []
myList = os.listdir(path)
# print(myList)  # List the filename in the directory of path

for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append((curImg))
    className.append(os.path.splitext(cl)[0])


# print(className)  # List the filename value from the path without extension

def enc(images):
    encodelist = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodelist.append(encode)
    return encodelist


# print(type(enc(images)[0]))

# print(enc(images))

# student_details = pd.read_csv(r'Student_encoded_data.csv', index_col=0)
# print(student_details)
#
# if any( (enc(images)==x).all for x in np.array(student_details.values.tolist())):
#     print("True")

students_dataframe = pd.DataFrame(enc(images), index=className)

# print(students_dataframe)

students_dataframe.to_csv(r'Student_encoded_data.csv')
print("All images have been encoded and saved")
print('--------------------------------------------------------------')
print(students_dataframe.head())

