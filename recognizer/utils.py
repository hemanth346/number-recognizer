from recognizer import device, CUDA
import numpy as np
import matplotlib.pyplot as plt

from torchsummary import summary
import csv

__telephone_directory = {}

def network_summary(network, input_size=(1, 28, 28), device=device):
    """
    Show summary of given network
    """
    print(device)
    model = network().to(device)
    print(summary(model, input_size=input_size, device=device.type))

def show_img(image):
    plt.imshow(image, cmap='gray')
    plt.show()


def view_classify(image, cls_probs):
    '''
    Function for viewing an image and it's predicted classes.
    '''
    if CUDA:
        cls_probs = cls_probs.data.cpu().numpy().squeeze()
    else:
        cls_probs = cls_probs.data.numpy().squeeze()
    fig, (ax1, ax2) = plt.subplots(figsize=(6,9), ncols=2)
    ax1.imshow(image, cmap='gray')
    ax1.axis('off')
    ax2.barh(np.arange(10), cls_probs)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(10))
    ax2.set_yticklabels(np.arange(10))
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)
    plt.tight_layout()
    plt.show()

def load_directory(file='telephone_directory.csv'):
    with open(file, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in reader:
            __telephone_directory[row[1]] = row[0]
        return __telephone_directory
