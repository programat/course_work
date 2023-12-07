
import cv2
from src.strategies import detection_strategy
from src.strategies import pose_processor_strategy
from src.strategies import angle_calculation_strategy

if __name__ == '__main__':
    detector = detection_strategy.YOLOStrategy(320, r'src/models/weights/yolov8s-pose.pt')

    detector.create_model()

    angle = angle_calculation_strategy.Angle2DCalculation()

    pose_pr = pose_processor_strategy.SquatsProcessor(detector, angle)

    vid = cv2.VideoCapture('/Users/egorken/Downloads/How to bodyweight squat.mp4')
    # vid = cv2.VideoCapture(1)
    while True:
        _, frame = vid.read()
        try:
            pose_pr.process(frame)
        except Exception as ex:
            print(ex)
        annotated_frame = detector.process_frame(frame)
        # res_coord = detector.get_coordinates()
        # print(res_coord)

        cv2.imshow('test', annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vid.release()
    cv2.destroyAllWindows()