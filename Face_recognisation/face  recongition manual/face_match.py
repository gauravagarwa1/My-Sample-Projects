import cv2
import face_recognition
import final_attendance


known_face_encodings=[]
known_face_names=[]

faces_recognised= set()


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

cam=cv2.VideoCapture(0)

while True:
    ret,frame=cam.read()

    face_locations=face_recognition.face_locations(frame)
    face_encodings=face_recognition.face_encodings(frame,face_locations)

    for (top, right, bottom, left), face_encoding in zip (face_locations, face_encodings):
        matches=face_recognition.compare_faces(known_face_encodings, face_encoding)
        name="Unknown"

        if True in matches:
            first_match_index=matches.index(True)
            name=known_face_names[first_match_index]
            faces_recognised.add(name)

        cv2.rectangle(frame,(left,top),(right,bottom), (0,0,255), 2)
        cv2.putText(frame, name,(left,top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)

    cv2.imshow("face screening", frame)

    if cv2.waitKey(1) & 0xFF == ord ('q'):
        final_attendance.give_attendance(faces_recognised)
        break

cam.release()

