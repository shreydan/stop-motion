import cv2

def detect_edges(frame):

    edges = cv2.Canny(frame,100,200)