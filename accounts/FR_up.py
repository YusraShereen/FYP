import face_recognition

def recognize(uploaded_img, nic_img):

  selfie = face_recognition.load_image_file(uploaded_img)
  up_face_encoding = face_recognition.face_encodings(selfie)[0]


  nic_pic= face_recognition.load_image_file(nic_img)

  nic_face_encoding = face_recognition.face_encodings(nic_pic)[0]

  results = face_recognition.compare_faces([up_face_encoding], nic_face_encoding)

  face_distances = face_recognition.face_distance([nic_face_encoding,nic_face_encoding], up_face_encoding)

  return face_distances[0]
