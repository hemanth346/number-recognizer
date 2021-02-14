import cv2
from recognizer.processor import *
    
def detect_blobs(grayImage):
    mser = cv2.MSER_create(_delta=1) # Create MSER object
    # https://stackoverflow.com/a/57623749/7445772
    regions, boundingBoxes = mser.detectRegions(grayImage) # detect regions in gray scale image
    boundingBoxes = non_max_suppression_fast(boundingBoxes, overlapThresh=0.2)
    return boundingBoxes #x, y, w, h 

def draw_bboxes(image, bboxes, save=False):
    for box in bboxes:
            x, y, w, h = box;
            cv2.rectangle(image, (x, y), (x+w, y+h), color=(0, 255, 0), thickness=1)
