import time

from ultralytics import YOLO
import cv2
import numpy as np


def findAngleAlt(img, p1, p2, p3, coord, draw=True):
    # landmarks
    x1, y1 = coord[p1]
    x2, y2 = coord[p2]
    x3, y3 = coord[p3]

    import math
    angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
    if angle < 0:
        # angle += 360
        angle = abs(angle)
    # print(angle)

    if draw:
        cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
        cv2.line(img, (x2, y2), (x3, y3), (255, 255, 255), 3)
        cv2.circle(img, (x1, y1), 9, (255, 0, 0), cv2.FILLED)
        cv2.circle(img, (x2, y2), 9, (255, 0, 0), cv2.FILLED)
        cv2.circle(img, (x3, y3), 9, (255, 0, 0), cv2.FILLED)
        cv2.putText(img, str(int(angle)), (x2 - 20, y2 + 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 3)

    return angle

def findAngle(img, p1,p2,ref_pt, coord, draw=True):
    p1 = coord[p1]
    p2, ref_pt = coord[ref_pt], coord[p2]
    p1_ref = p1 - ref_pt
    p2_ref = p2 - ref_pt

    cos_theta = (np.dot(p1_ref, p2_ref)) / (1.0 * np.linalg.norm(p1_ref) * np.linalg.norm(p2_ref))
    theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))

    degree = int(180 / np.pi) * theta

    if draw:
        cv2.line(img, p1, ref_pt, (255, 255, 255), 3)
        cv2.line(img, ref_pt, p2, (255, 255, 255), 3)
        cv2.circle(img, p1, 9, (255, 0, 0), cv2.FILLED)
        cv2.circle(img, ref_pt, 9, (255, 0, 0), cv2.FILLED)
        cv2.circle(img, p2, 9, (255, 0, 0), cv2.FILLED)
        cv2.putText(img, str(int(degree)), (ref_pt[0] - 20, ref_pt[1] + 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 3)

    return int(degree)


def main() -> None:
    # model = YOLO('yolov8s-pose.pt')
    # # results = model(source="/Users/egorken/Downloads/IMG_5497.jpeg", device='cpu')
    # results = model(source=1, show=True, device='cpu')0
    # array = results[0].plot()
    # print(results[0].keypoints.xy.numpy())
 
    vid = cv2.VideoCapture(1)
    # vid = cv2.VideoCapture(r'/Users/egorken/Downloads/Exercise Tutorial - Squat.mp4')
    # vid = cv2.VideoCapture(r'/Users/egorken/Downloads/x2mate.com-How to do a Dumbbell Hammer Curl.mp4')
    model = YOLO('../weights/yolov8x-pose.pt')
    count = 0
    dir = 0
    pTime = 0


    while True:
        ret, frame = vid.read()
        # frame = cv2.resize(frame, (1280, 720))
        # cv2.imshow('frame', frame)

        # img = cv2.imread(r'/Users/egorken/Downloads/squats.png')
        results = model(frame, verbose=False, device='cpu', imgsz=320)
        # results = model(source="/Users/egorken/Downloads/squats.png", device='cpu', verbose=False)  # verbose is used for turning off printing results
        annotated_frame = results[0].plot(labels=False, boxes=False)
        res_coord = [r.keypoints.xy.to(int).numpy() for r in results]

        if len(res_coord) != 0:
            try:
                # left arm
                angle = findAngle(annotated_frame, 5, 7, 9, res_coord[0][0])
                # per = np.interp(angle, (210, 310), (0, 100))
                # bar = np.interp(angle, (210, 310), (650, 100))
                per = np.interp(angle, (65, 160), (100, 0))
                bar = np.interp(angle, (65, 160), (100, 650))

                # check for curls
                color = (255, 0, 0)
                if per == 100:
                    color = (0, 255, 0)
                    if dir == 0:
                        count += .5
                        dir = 1
                if per == 0:
                    color = (0, 255, 0)
                    if dir == 1:
                        count += .5
                        dir = 0

                print(f"\rleft arm: angle {angle}, % {per}, count {count}", end='')

                # drawing bar
                cv2.rectangle(annotated_frame, (1100, 100), (1175, 650), (255, 255, 255), 3)
                cv2.rectangle(annotated_frame, (1100, int(bar)), (1175, 650), color, cv2.FILLED)
                cv2.putText(annotated_frame, f'{int(per)}%', (1100, 700), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)

                # drawing curl
                cv2.rectangle(annotated_frame, (0,650), (70, 720), (255,255,255), cv2.FILLED)
                cv2.putText(annotated_frame, f'{int(count)}', (10,700), cv2.FONT_HERSHEY_PLAIN, 3, (0,0,0), 5)


                # right arm
                findAngle(annotated_frame, 6, 8, 10, res_coord[0][0])

                # # left leg
                # findAngle(annotated_frame, 11, 13, 15, res_coord[0][0])
                # # right leg
                # findAngle(annotated_frame, 12, 14, 16, res_coord[0][0])
            except:
                pass

        cTime = time.time()
        fps = 1/(cTime - pTime)
        pTime = cTime
        cv2.putText(annotated_frame, f'fps: {int(fps)}', (1160,60), cv2.FONT_HERSHEY_PLAIN, 1.2, (255,255,255), 2)

        cv2.imshow('frame', annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    print(f"\n{cv2.getWindowImageRect('frame')}")
    vid.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
