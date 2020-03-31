import numpy as np
import cv2
import time
import random
import matplotlib.pyplot as plt

# Definition for Face detection
face_cascade = cv2.CascadeClassifier('Haar/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('Haar/haarcascade_eye.xml')

# Definition for face tracking
(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
tracker_type = tracker_types[7]
previous_face_size = 0
max_face_size = 28000
min_face_size = 10000


def create_tracker(type):
    if int(minor_ver) < 3:
        tk = cv2.Tracker_create(type)
    else:
        if type == 'BOOSTING':
            tk = cv2.TrackerBoosting_create()
        if type == 'MIL':
            tk = cv2.TrackerMIL_create()
        if type == 'KCF':
            tk = cv2.TrackerKCF_create()
        if type == 'TLD':
            tk = cv2.TrackerTLD_create()
        if type == 'MEDIANFLOW':
            tk = cv2.TrackerMedianFlow_create()
        if type == 'GOTURN':
            tk = cv2.TrackerGOTURN_create()
        if type == 'MOSSE':
            tk = cv2.TrackerMOSSE_create()
        if type == "CSRT":
            tk = cv2.TrackerCSRT_create()
    return tk


def check_center(face_info, img):
    face_info = face_info[0]
    h, w = img.shape[0], img.shape[1]
    statify_flg = False
    x_min = w / 5 * 2
    x_max = w / 5 * 3
    y_min = h / 5 * 2
    y_max = h / 5 * 3

    bbox_center_x = face_info[0] + (face_info[2] / 2)
    bbox_center_y = face_info[1] + (face_info[3] / 2)

    if x_min < bbox_center_x < x_max and y_min < bbox_center_y < y_max:
        statify_flg = True

    return statify_flg


def validate_face_bouding(face_info, img):
    face_valid = []
    h, w = img.shape[0], img.shape[1]
    x_min = w / 4
    x_max = w / 4 * 3
    global previous_face_size
    global min_face_size
    global max_face_size

    for face in face_info:
        valid_flg = False
        # 1. Condition center
        bbox_center_x = face[0] + face[2]/2
        if x_min < bbox_center_x < x_max:
            valid_flg = True

        # 2. Condition for changing size of face
        current_face_size = face[2] * face[3]
        if previous_face_size != 0:
            different = (np.abs(current_face_size - previous_face_size) / previous_face_size) * 100
            if different < 30:
                valid_flg = True
                previous_face_size = current_face_size
            else:
                valid_flg = False
        else:
            if min_face_size < current_face_size < max_face_size:
                previous_face_size = current_face_size
                valid_flg = True

        # Add other condition

        if valid_flg:
            face_valid.append(face)

    return face_valid


def draw_face(face_info, img):
    img_output = img.copy()
    for (x, y, w, h) in list(face_info):
        img_output = cv2.rectangle(img_output, (x, y), (x + w, y + h), (255, 0, 0), 2)
        # roi_gray = gray[y:y + h, x:x + w]
        # roi_color = img_output[y:y + h, x:x + w]
        # eyes = eye_cascade.detectMultiScale(roi_gray)
        # for (ex, ey, ew, eh) in eyes:
        #     cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
    return img_output


def draw_text(face_info, temperature, img, id):
    if temperature < 37:
        color = (0, 255, 0)
    else:
        color = (0, 0, 255)
    for (x, y, w, h) in face_info:
        cv2.putText(img, 'Face {}: {}*C'.format(str(id), str(temperature)), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)


def get_temparature():
    temp = random.randrange(350, 400, 5) / 10
    return temp


def check_out_range(face_info, img):
    flag_out_range = False
    im_h, im_w = img.shape[0], img.shape[1]

    for face in face_info:
        left_x, left_y, right_x, right_y = bbox_center_x = face[0], face[1], face[0] + face[2], face[1] + face[3]
        if left_x <= 100 or left_y <= 10 or right_x >= (im_w - 100) or right_y >= (im_h - 10):
            flag_out_range = True

    return flag_out_range


def remove_face(face_info):
    valid_face = []
    max_id = 0
    max_face = 0
    global min_face_size
    global max_face_size

    for face_id, face in enumerate(face_info):
        current_face_size = face[2] * face[3]
        if min_face_size <= current_face_size <= max_face_size:
            if current_face_size > max_face:
                max_face = current_face_size
                max_id = face_id

    valid_face = [face_info[max_id]]

    return valid_face

# Read from video file
# video_pth = '/home/tducnguyen/Downloads/Video/Springfield Armory Hellcat Micro Compact Pistol Review.mp4'
# cap = cv2.VideoCapture(video_pth)

# Read from camera
cap = cv2.VideoCapture(0)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
stored_file_name = 0
out = cv2.VideoWriter('Haar/streams/video_file_{}.avi'.format(stored_file_name), cv2.VideoWriter_fourcc(*'XVID'), 30, (frame_width, frame_height))

file_range = 10*60*30  # total minutes * 60s * 30 fps = total frames
counter = 0
previous_face = []
count_dispear = 0
people_id = 0
new_people_allow = True

while True:
    # t = time.time()
    ret, frame = cap.read()
    fr_h, fr_w = frame.shape[0], frame.shape[1]

    if ret:
        counter += 1
        # print(counter)
        if counter >= file_range:
            stored_file_name += 1
            out = cv2.VideoWriter('Haar/streams/video_file_{}.avi'.format(stored_file_name), cv2.VideoWriter_fourcc(*'XVID'), 30, (frame_width, frame_height))
            counter = 0

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect face
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        # Filter the face bounding
        faces = validate_face_bouding(faces, frame)
        if len(faces) > 1:
            # Remove redundant face
            faces = remove_face(faces)
            count_dispear = 0

        if len(faces) == 1:
            # Store faces extracted for tracking
            previous_face = faces
            count_dispear = 0

            print('Inferred face')
            if new_people_allow:
                # check center
                if check_center(faces, frame):
                    people_id += 1
                    new_people_allow = False

        elif len(faces) == 0:
            count_dispear += 1
            # create tracking faces based stored feature
            # if previous_face != []:
            #     tracker = create_tracker(tracker_type)
            #     status_init = tracker.init(gray, tuple(previous_face[0]))
            #     # print('status_init: ', status_init)
            #     ok, faces_track = tracker.update(gray)
            #     if ok:
            #         faces = [[int(i) for i in faces_track]]
            #     else:
            #         print('Can not tracking')
            #
            #     print('Tracked frame')

        # Check whether face can not detect by Haar cv and tracking out of frame
        if not new_people_allow:
            if check_out_range(faces, frame):
                print('Out range')
                new_people_allow = True
                previous_face = []

        if count_dispear > 120:
            faces = []
            previous_face = []
            print('Object disappeared')
            new_people_allow = True


        # Read temperature
        temp = get_temparature()
        if len(faces) > 0 and check_center(faces, frame):
            frame = draw_face(faces, frame)
            draw_text(faces, temp, frame, people_id)


        # Write the frame into the file
        out.write(frame)

        # print(time.time() - t)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# When everything done, release the video capture and video write objects
cap.release()
out.release()

# Closes all the frames
cv2.destroyAllWindows()
