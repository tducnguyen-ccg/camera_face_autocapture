import numpy as np
import cv2
import time
import random
import matplotlib.pyplot as plt
from state import State
from datetime import datetime, date

# Definition for Face detection
face_cascade = cv2.CascadeClassifier('Haar/haarcascade_frontalface_default.xml')

# State
_last_state = State.Unknown
_state = State.Unknown


def write_log(text):
    log = open("logs/log-" + str(date.today()) + ".txt", 'a')
    log.write(str(datetime.now().isoformat()) + ':    ' + text + '\n')
    log.close()


def non_max_suppression_fast(boxes, overlapThresh = 0.2):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []
    boxes = np.array(boxes)
    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions

    boxes = boxes.astype("float")
    # initialize the list of picked indexes
    pick = []
    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]
    y2 = boxes[:, 1] + boxes[:, 3]
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = np.abs((x2 - x1 + 1) * (y2 - y1 + 1))
    idxs = np.argsort(y2)
    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]
        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
            np.where(overlap > overlapThresh)[0])))
    # return only the bounding boxes that were picked using the
    # integer data type
    boxes = boxes[pick].astype("int")

    # merge all boxes:
    boxes = [[min(boxes[:, 0]), min(boxes[:, 1]), max(boxes[:, 2]), max(boxes[:, 3])]]

    return boxes


def check_center(box, img):
    box = box[0]
    h, w = img.shape[0], img.shape[1]
    statify_flg = False
    x_min = w / 5 * 2
    x_max = w / 5 * 3
    y_min = h / 5 * 2
    y_max = h / 5 * 3

    bbox_center_x = box[0] + (box[2] / 2)
    bbox_center_y = box[1] + (box[3] / 2)

    if x_min < bbox_center_x < x_max and box[2]*box[3] >= 30000:
        statify_flg = True

    return statify_flg


def draw_temperature(temperature, img, id, coor):
    img_output = img.copy()
    if temperature < 37:
        color = (0, 255, 0)
    else:
        color = (0, 0, 255)

    img_output = cv2.putText(img_output, 'Person {}: {}*C'.format(str(id), str(temperature)), coor, cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    return img_output


def draw_text(text, img, coor):
    img_out = img.copy()
    img_out = cv2.putText(img_out, text, coor, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    return img_out


def get_temparature():
    temp = random.randrange(350, 400, 5) / 10
    return temp



def contour_filter(contour):
    valid_contour = []
    for c in contour:
        x, y, w, h = cv2.boundingRect(c)
        # Condition 1: size not too large
        area_cond = cv2.contourArea(c) >= 1000
        # print(cv2.contourArea(c))
        # Condition 2: not too long
        # length_cond = w < 250
        # Condition 3: ratio between height and width
        # ratio = np.abs(w - h)/h < 0.9
        if area_cond:
            valid_contour.append(c)

    return valid_contour


def switch_state(state):
    global _state
    _last_state = _state
    _state = state


def motion_detection(fr, pres_fr):
    d = cv2.absdiff(fr, pres_fr)
    grey = cv2.cvtColor(d, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grey, (5, 5), 0)
    ret, th = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    dilation = cv2.dilate(th, np.ones((5, 5), np.uint8), iterations=3)

    # erosion = cv2.erode(dilation, np.ones((3, 3), np.uint8), iterations=3)
    _, c, _ = cv2.findContours(dilation, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    c = list(c)
    c = contour_filter(c)
    boxs = []
    for c_ in c:
        x, y, w, h = cv2.boundingRect(c_)
        if w * h > 10000:  # Remove small object
            # print(w * h)
            boxs.append([x, y, w, h])

    boxs = non_max_suppression_fast(boxs)

    return c, boxs


def frame_is_different(frame_1, frame_2):
    global interest_region
    flag_diff = False

    d = cv2.absdiff(frame_1, frame_2)
    grey = cv2.cvtColor(d, cv2.COLOR_BGR2GRAY)

    inreset_value = (grey * interest_region)

    diff_sum = np.sum(inreset_value)
    if diff_sum > 800000:
        # print(diff_sum)
        flag_diff = True

    return flag_diff


def stop_modetion_detect(fr, pres_fr):
    global interest_region
    stop_flg = False

    d = cv2.absdiff(fr, pres_fr)
    grey = cv2.cvtColor(d, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grey, (5, 5), 0)
    ret, th = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    # dilation = cv2.dilate(th, np.ones((5, 5), np.uint8), iterations=3)

    inreset_value = (th * interest_region)
    # print(np.sum(inreset_value))
    if np.sum(inreset_value) <= 500000:
        stop_flg = True
    # print(stop_flg)

    return stop_flg


def state_init():
    # Initialize
    global cap
    global frame
    global frame_rs
    global face_cascade
    global faces
    global temp
    global people_id
    global previous_frame
    global interest_region
    global state_still_counter
    global timer
    global display_counter
    global allow_next_people
    global motion_counter
    global timeout
    global video_mask
    global background_frame
    global background_counter
    global background_timer

    cap = cv2.VideoCapture(0)
    _, frame = cap.read()
    previous_frame = frame.copy()
    frame_rs = frame.copy()
    background_frame = frame.copy()

    h, w = frame.shape[0], frame.shape[1]
    interest_region = np.zeros((h, w))
    x_min = int(w / 10 * 3.5)
    x_max = int(w / 10 * 6.5)
    y_min = int(h / 5 * 1.5)
    y_max = int(h / 6 * 4.5)

    for r in range(h):
        for c in range(w):
            if y_min < r < y_max and x_min < c < x_max:
                interest_region[r, c] = 1

    temp = 0
    people_id = 0
    state_still_counter = 0
    timer = time.time()
    display_counter = 0
    motion_counter = 0
    allow_next_people = True
    timeout = 0
    video_mask = interest_region.copy()
    video_mask[video_mask == 0] = 0.7
    background_counter = 0
    background_timer = 0

    switch_state(State.Standby)


def mask_frame(frame, video_mask):
    frame_out = frame.copy()

    for i in range(3):
        frame_out[:, :, i] = frame_out[:, :, i] * video_mask

    return frame_out


def state_standby():
    # Waiting to change to other state
    global cap
    global frame
    global previous_frame
    global frame_rs
    global temp
    global people_id
    global timer
    global video_mask
    global background_frame
    global background_counter

    flag_next_state = False

    ret, frame = cap.read()
    frame_rs = frame.copy()
    # Background not change
    if background_counter >= 10 * 60 * 30:  # 60s , each second 30 fps /iteration
        background_frame = frame.copy()
        background_counter = 0
        print('Background changed')
        write_log('Background changed')
    background_counter += 1

    # Compare frame with background every 2 seconds
    if background_counter % (2 * 30) == 0:
        diff = frame_is_different(background_frame, frame)
        if diff:
            flag_next_state = True
            write_log('Detect different with background')

    motion_info, boxes = motion_detection(frame, previous_frame)

    cv2.drawContours(frame_rs, motion_info, -1, (0, 255, 0), 2)
    # print(len(boxes))
    if len(boxes) > 0:
        for box in boxes:
            x, y, w, h = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            cv2.rectangle(frame_rs, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Check if bounding is in center
        if check_center(boxes, frame_rs):
            flag_next_state = True

    if flag_next_state:
        switch_state(State.InformationCapture)
        frame_rs = draw_text('Please stand still for a second !', frame_rs, (100, 50))
        background_counter = 0

        # Capture image:
        # Start timer:
        timer = time.time()

    frame_rs = mask_frame(frame_rs, video_mask)
    previous_frame = frame
    print('state_standby')
    write_log('state_standby')


def state_informationCapture():
    # Waiting to read the
    global frame_rs
    global interest_region
    global cap
    global frame
    global previous_frame
    global state_still_counter
    global motion_counter
    global timer
    global temp
    global display_counter
    global allow_next_people
    global people_id
    global video_mask

    ret, frame = cap.read()
    frame_rs = frame.copy()
    frame_rs = draw_text('Please stand still for a second !', frame_rs, (100, 50))

    stop_flg = stop_modetion_detect(frame, previous_frame)
    if stop_flg:
        if (time.time() - timer) >= 0.2:
            timer = time.time()
            state_still_counter += 1
        # state_still_counter = time.time() - timer
    else:
        timer = time.time()
        state_still_counter = 0
    # Check if not motion in the region of interest

    # frame_rs = draw_text('{}s'.format(state_still_counter), frame_rs, (280, 100))

    if allow_next_people:
        people_id += 1
        allow_next_people = False

    if stop_flg and state_still_counter >= 1:  # Time to waiting the machine read temperature
        if display_counter == 0:
            display_counter += 1
            # Capture temperature
            temp = get_temparature()
            frame_rs = draw_temperature(temp, frame_rs, people_id, (180, 450))
        else:
            time.sleep(1)
            display_counter = 0
            frame_rs = draw_temperature(temp, frame_rs, people_id, (180, 450))
            allow_next_people = False
            motion_counter = 0
            state_still_counter = 0
            switch_state(State.MotionTracking)

    frame_rs = mask_frame(frame_rs, video_mask)
    previous_frame = frame


def state_motion_tracking():
    global frame_rs
    global interest_region
    global cap
    global frame
    global previous_frame
    global timer
    global motion_counter
    global allow_next_people
    global state_still_counter
    global timeout
    global video_mask


    ret, frame = cap.read()
    frame_rs = frame.copy()

    # Waiting until object disappear in the center
    stop_flg = stop_modetion_detect(frame, previous_frame)

    if not stop_flg:
        # check if center is clean
        allow_next_people = True
        motion_counter += 1
        print('motion ', motion_counter)
        write_log('motion {}'.format(motion_counter))
    else:
        timeout += 1
        if allow_next_people and motion_counter > 2:
            state_still_counter += 1
            if state_still_counter >= 10:
                switch_state(State.Standby)
                motion_counter = 0
                state_still_counter = 0
        if timeout >= 150:
            timeout = 0
            switch_state(State.Standby)
            motion_counter = 0
            state_still_counter = 0

    # Waiting until object disappear in the conner

    frame_rs = mask_frame(frame_rs, video_mask)
    previous_frame = frame
    print('state_motion_tracking')
    write_log('state_motion_tracking')


def loop():
    global _state
    switcher = {
        State.Init: state_init,
        State.Standby: state_standby,
        State.InformationCapture: state_informationCapture,
        State.MotionTracking: state_motion_tracking

    }
    func = switcher.get(_state)
    return func()


def main():
    switch_state(State.Init)

    while True:
        loop()
        cv2.imshow('frame', frame_rs)
        # check face detected on center
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    main()

