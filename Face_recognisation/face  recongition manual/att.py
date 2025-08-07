import cv2
import face_recognition
# import final_attendance
import csv
from datetime import datetime

# load known face encodings and names 
known_face_encodings=[]
known_face_names=[]

faces_recognised= set()

# load known face and their names
known_person1_image= face_recognition.load_image_file("harsh.jpg")
known_person2_image= face_recognition.load_image_file("gaurav.jpg")
known_person3_image= face_recognition.load_image_file("srijan.jpg")
known_person4_image= face_recognition.load_image_file("dev.jpg")

known_person1_encoding=face_recognition.face_encodings(known_person1_image)[0]
known_person2_encoding=face_recognition.face_encodings(known_person2_image)[0]
known_person3_encoding=face_recognition.face_encodings(known_person3_image)[0]
known_person4_encoding=face_recognition.face_encodings(known_person4_image)[0]

known_face_encodings.append(known_person1_encoding)
known_face_encodings.append(known_person2_encoding)
known_face_encodings.append(known_person3_encoding)
known_face_encodings.append(known_person4_encoding)


known_face_names.append("harsh")
known_face_names.append("gaurav")
known_face_names.append("srijan")
known_face_names.append("Devansh")

students = known_face_names.copy()

#current date and time
now =datetime.now()
current_date=now.strftime("%Y-%m-%d")

#open csv file 
f=open(f"{current_date}.csv", "w+", newline="")
lnwritter=csv.writer(f)

# initialize webcam for face 
cam=cv2.VideoCapture(0)

while True:
    # capture face frame by frame
    ret,frame=cam.read()


    # find all the face location in current frame
    face_locations=face_recognition.face_locations(frame)
    face_encodings=face_recognition.face_encodings(frame,face_locations)


    # here we run a loop (to detect or) for each face found in the frame
    for (top, right, bottom, left), face_encoding in zip (face_locations, face_encodings):
        # here we match the faces 
        matches=face_recognition.compare_faces(known_face_encodings, face_encoding)
        name="Unknown"

        if True in matches:
            first_match_index=matches.index(True)
            name=known_face_names[first_match_index]
            faces_recognised.add(name)

        # draw the box around the face and show name
        cv2.rectangle(frame,(left,top),(right,bottom), (0,0,255), 2)
        cv2.putText(frame, name,(left,top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)

        # for attendance
        if name in students:
                students.remove(name)
                current_time = now.strftime("%H-%M-%S")
                lnwritter.writerow([name, current_time])
        
    # display the resulting frame
    cv2.imshow("face screening", frame)

    # print the attendance
    def give_attendance(names):
        print(names)
        print(current_date)


# break the loop by pressing "q"
    if cv2.waitKey(1) & 0xFF == ord ('q'):
        give_attendance(faces_recognised)
        break

# close the webcam and opencv window
cam.release()
cv2.destroyAllWindows

