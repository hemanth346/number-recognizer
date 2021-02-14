import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from matplotlib import pyplot as plt

import argparse
import csv

from recognizer import device
from recognizer.processor import preprocess
from recognizer.utils import * 

# use_cuda = False #torch.cuda.is_available()
# device = torch.device("cuda" if use_cuda else "cpu")

__telephone_directory = {}

_transforms = transforms.Compose([
                                    transforms.Resize((28, 28)), #justincase
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))
                                    ])



def parse_args():
    parser = argparse.ArgumentParser(description='Detect mobile number')

    parser.add_argument('--model',
            required=True,
            help='path to saved model state',
            type=str
            )

    parser.add_argument("-i",
            "--image",
            required=True,
            help="path to image"
            )    
    args = parser.parse_args() # use namespace as is
    return args

def load_model(path):
    model = torch.load(path)
    model.eval()
    return model

def detect_number(filename, model, save=False, device=device):
    # grayImage = cv2.imread(filename, cv2.IMREAD_GRAYSCALE) # read as gray scale
    image = cv2.imread(filename)
    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    bboxes = preprocess.detect_blobs(grayImage)
    number = ''
    ksize, threshold = 1, 120
    for idx, box in enumerate(bboxes):
        x, y, w, h = box
        roi = grayImage[y:y+h, x:x+w]
        roi = cv2.resize(roi, (28,28))
        show_img(roi)
        blurred = cv2.boxFilter(roi, -1, (ksize,ksize) ,normalize=False)      
        _, thresh = cv2.threshold(blurred,threshold,255,cv2.THRESH_BINARY)
        # thresh = cv2.adaptiveThreshold(blurred,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY, blockSize=11, C=2) # make the image similar to train data
        # load image as tensor
        input_image = Image.fromarray(thresh)
        input_image = _transforms(input_image).unsqueeze(0)
        # prediction
        output = model(input_image.to(device))
        _, pred = output.max(dim=1, keepdim=True)
        # Output of the network are log-probabilities, need to take exponential for probabilities
        cls_probabilities = torch.exp(output)
        view_classify(thresh, cls_probabilities)
        print('probabilities : ', cls_probabilities,'prediction : ', pred.item())
        number += str(pred.item())
    return number

def load_directory(file='telephone_directory.csv'):
    with open(file, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in reader:
            __telephone_directory[row[1]] = row[0]
        return __telephone_directory

def find_name(number):
    if not __telephone_directory:
        load_directory()
    return __telephone_directory.get(number, 'Number not found in directory')

# def main():
#     args = parse_args()
#     # path = 'recognizer\weights\entire_model.pt'
#     model = load_model(path=args.model)
#     filename = args.image
#     predicted = detect_number(filename, model)
#     print('Identified number  is {0}'.format(predicted))
#     print('Checking for name in telephone directory...')
#     find_name(predicted)

# if __name__ == '__main__':
#     main()

