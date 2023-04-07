import cv2
from ultralytics import YOLO
import cvzone
model = YOLO('best-bd.pt')
class Detector:
    def __init__(self):
        self.classNames = ["COT"]
        self.model = YOLO('best-bd.pt')
        self.obj_boxes = []
        self.obj_classes = []
        self.obj_centers = []


    def detect_object(self, bgr_frame):
        res = self.model(bgr_frame, stream=True)
        self.obj_boxes = []
        self.obj_classes = []
        self.obj_centers = []
        self.obj_distances = []
        for r in res:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1
                cls = int(box.cls[0])
                curClass = self.classNames[cls]
                if curClass == "COT":
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2
                    self.obj_centers.append((cx, cy))
                    self.obj_boxes.append((x1, y1, w, h))
                    self.obj_classes.append(curClass)
        return self.obj_boxes, self.obj_centers, self.obj_classes

    def draw_object_info(self, bgr_frame, depth_frame):
        for i in range(len(self.obj_boxes)):
            box = self.obj_boxes[i]
            cx, cy = self.obj_centers[i]
            cls = self.obj_classes[i]
            distance = depth_frame[cy, cx]
            cvzone.cornerRect(bgr_frame, box)
            cvzone.putTextRect(bgr_frame, f"({cx},{cy}): {distance}mm", (max(box[0], 0), max(35, box[1])), 2, 2)
            cv2.circle(bgr_frame, (cx, cy), 5, (255, 0, 255), -1)

        return bgr_frame



# cap = cv2.VideoCapture(0)
# model = YOLO('yolov8n.pt')
#
# cap.set(3, 1280)
# cap.set(4, 680)
#
# classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
#               "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
#               "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
#               "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
#               "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
#               "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
#               "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
#               "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
#               "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
#               "teddy bear", "hair drier", "toothbrush"
#               ]
#
# while True:
#     success, image = cap.read()
#     res = model(image, stream=True)
#     for r in res:
#         boxes = r.boxes
#         for box in boxes:
#             x1, y1, x2, y2 = box.xyxy[0]
#             ww = math.sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1))
#             x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
#             w, h = x2 - x1, y2 - y1
#             cls = int(box.cls[0])
#             conf = math.ceil(box.conf[0] * 100) / 100
#             curClass = classNames[cls]
#             if curClass == "cell phone":
#                 cvzone.cornerRect(image, (x1, y1, w, h))
#                 f = 530
#                 W = 16.882328
#                 d = int((W * f) / ww)
#                 cvzone.putTextRect(image, f"{d} cm", (max(x1, 0), max(35, y1)), 2, 2)
#     cv2.imshow("Image", image)
#     cv2.waitKey(1)