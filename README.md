# Number Recognizer
> ### App to detect and recognize mobile number on a sheet of paper using laptop webcam

<br>
Task is to build an application that recognizes hand written telephone numbers and matches it to a name from pre-loaded telephone directory

    > Supposed to be done in a weekend. Unfortunately, due to other circumstances I'm unavailable on saturday and was only able to do some literature review and no experiment, so effectively I only had one day.! 

### Assumptions and rationale:
1. The model is created with the assumption that **only 1 mobile number** will be in a frame of white sheet.
    - Complexity of the expected input data(i.e. Handwritten numbers) is not known, hence used simplest as POC.
  
2. Used ML model used only for digit classification, rather than using sequence or object detection model keeping edge device in mind
    - Heavylifting to be done by data processing pipelines using OpenCV.
3. Using flatfile(csv) as my database


## Task:

Given task can be subdivided into 
1. Localization task
   1. Digit localization
   2. Whole number localization
2. Classification task

## Approach:

### 1. Localization i.e. Finding digits regions:

Localization can be achieved either using object detection models or by image preprocessing followed by segmenting the digits. Since there is limit on compute available we can't go with deep learning models for object detection. 

For image preprocessing, I've tried below methods

- **Method 1**
   1. Convert to grayscale
   2. Use edge detector to find the edges
   3. Find digits using contour finder
   4. Extract ROI, resize it and send for classfication

- **Method 2** 
   1. Convert to grayscale
   2. Use Maximally Stable Extremal Regions(MSER) method to find blobs regions
       - Tuning delta value helped a lot
   3. Apply non maximum suppression(NMS) to get bounding box for the likely digits
   4. Extract ROI, resize it and send for classfication

 - Limitations:
   - Where multiple digits are grouped in single bbox when numbers are combined
   - Contours/blobs can be formed by anything in the frame not necessarily digits

![Image with bounding boxes](./tmp/1_bboxes.jpg 'Digits identified')

### 2. Classification i.e. Identifying digits:

For classification, used MNIST dataset to train model as suggested. 

Two network architectures are made both achieving test accuracy of more than 99%

- **Network 1 : 7,758 Parameters**

<!-- abvdyl -->
Reached 99% test accuracy in less than 15 epochs. Since the model is very light light, the performance on real word data is not great.


[Colab Link(GPU)](https://colab.research.google.com/drive/1P2gTPNCk7QAoopsc_1dMaJkmnPdS-zGn?usp=sharing) - With class probability visualization

[Colab Link(CPU)](https://colab.research.google.com/drive/18WbohBCcaFsYnYFFPgE5F7gO3TxrUlLd)


- For each epoch training took 
  - less than _30 seconds_ on Tesla T4 16G GPU with _64 batch_size_ and 
  - around _120 seconds_ on CPU with _32 batch_size_ 
  
- Saved model is less than 100kB


```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 10, 28, 28]              90
       BatchNorm2d-2           [-1, 10, 28, 28]              20
           Dropout-3           [-1, 10, 28, 28]               0
              ReLU-4           [-1, 10, 28, 28]               0
            Conv2d-5           [-1, 10, 28, 28]             900
       BatchNorm2d-6           [-1, 10, 28, 28]              20
           Dropout-7           [-1, 10, 28, 28]               0
              ReLU-8           [-1, 10, 28, 28]               0
         MaxPool2d-9           [-1, 10, 14, 14]               0
           Conv2d-10           [-1, 10, 12, 12]             900
      BatchNorm2d-11           [-1, 10, 12, 12]              20
          Dropout-12           [-1, 10, 12, 12]               0
             ReLU-13           [-1, 10, 12, 12]               0
           Conv2d-14           [-1, 10, 10, 10]             900
      BatchNorm2d-15           [-1, 10, 10, 10]              20
             ReLU-16           [-1, 10, 10, 10]               0
           Conv2d-17             [-1, 10, 8, 8]             900
      BatchNorm2d-18             [-1, 10, 8, 8]              20
          Dropout-19             [-1, 10, 8, 8]               0
             ReLU-20             [-1, 10, 8, 8]               0
           Conv2d-21             [-1, 16, 6, 6]           1,440
      BatchNorm2d-22             [-1, 16, 6, 6]              32
          Dropout-23             [-1, 16, 6, 6]               0
             ReLU-24             [-1, 16, 6, 6]               0
           Conv2d-25             [-1, 16, 4, 4]           2,304
      BatchNorm2d-26             [-1, 16, 4, 4]              32
          Dropout-27             [-1, 16, 4, 4]               0
             ReLU-28             [-1, 16, 4, 4]               0
        AvgPool2d-29             [-1, 16, 1, 1]               0
           Conv2d-30             [-1, 10, 1, 1]             160
================================================================
Total params: 7,758
Trainable params: 7,758
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.61
Params size (MB): 0.03
Estimated Total Size (MB): 0.64
----------------------------------------------------------------
```

- **Network 2**
   
<!-- S10 -->
Increased parameter count by almost 100%, still network has to have more params to generalize better
Reached 99.5% test accuracy in 20 epochs with ~14k parameters 


[Colab Link(CPU)](https://colab.research.google.com/drive/10uFBIIWZ4DdQoCGnY0_K6S67zqJtBEDV?usp=sharing)

[Colab Link(GPU)](https://colab.research.google.com/drive/18yg9JR25Fxj1l00um430h4ttZEViEtPn?usp=sharing)

- For each epoch training took 
  - around _30 seconds_ on Tesla T4 16G GPU with _64 batch_size_ and 
  - around _120 seconds_ on CPU with _32 batch_size_ 
  
- Saved model is less than 100kB

```
  ----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 16, 26, 26]             144
              ReLU-2           [-1, 16, 26, 26]               0
       BatchNorm2d-3           [-1, 16, 26, 26]              32
           Dropout-4           [-1, 16, 26, 26]               0
            Conv2d-5           [-1, 32, 24, 24]           4,608
              ReLU-6           [-1, 32, 24, 24]               0
       BatchNorm2d-7           [-1, 32, 24, 24]              64
           Dropout-8           [-1, 32, 24, 24]               0
            Conv2d-9           [-1, 10, 24, 24]             320
        MaxPool2d-10           [-1, 10, 12, 12]               0
           Conv2d-11           [-1, 16, 10, 10]           1,440
             ReLU-12           [-1, 16, 10, 10]               0
      BatchNorm2d-13           [-1, 16, 10, 10]              32
          Dropout-14           [-1, 16, 10, 10]               0
           Conv2d-15             [-1, 16, 8, 8]           2,304
             ReLU-16             [-1, 16, 8, 8]               0
      BatchNorm2d-17             [-1, 16, 8, 8]              32
          Dropout-18             [-1, 16, 8, 8]               0
           Conv2d-19             [-1, 16, 6, 6]           2,304
             ReLU-20             [-1, 16, 6, 6]               0
      BatchNorm2d-21             [-1, 16, 6, 6]              32
          Dropout-22             [-1, 16, 6, 6]               0
           Conv2d-23             [-1, 16, 6, 6]           2,304
             ReLU-24             [-1, 16, 6, 6]               0
      BatchNorm2d-25             [-1, 16, 6, 6]              32
          Dropout-26             [-1, 16, 6, 6]               0
        AvgPool2d-27             [-1, 16, 1, 1]               0
           Conv2d-28             [-1, 10, 1, 1]             160
================================================================
Total params: 13,808
Trainable params: 13,808
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 1.06
Params size (MB): 0.05
Estimated Total Size (MB): 1.12
----------------------------------------------------------------
```




## **Limitations**:
  - Model has to classify the image into one of the 10 labels. Unlike modern object detection models where weightage is given for no objectess as well.
    
  - The model doesn't account for the order takes all blobs in the image and predict some number for it. 
    - High chances are false positives due to this

  - Single ROI can contain multiple digits in it which will fail the model utterly

  - Current pipeline fails in many scenarios where the input image has lot of frames. Preprocessing pipeline has to be tuned based on requirements.



## Further improvement:

- Use Street View House Numbers (SVHN) Dataset with rich features and annotated bounding boxes around digits in training data. Can increase deployment size drastically.
- Use sequence models/loss(?) to be able to identify the order when multiple numbers are present 

<!-- Expectations:
1. Real time inference using webcam
2. Should be portable and easy to setup
3. Should be light weight (can train on CPUs) 
-->

### Issues faced:
- Heavy tuning involved for preprocessing with OpenCV
- Lost lot of time as the ROI/digit image was not made similar to train data,i.e black over white background. Focused only getting boundaries/digits intact and missed to notice the issue until very late.
- Migrating working model from notebook into package and dependency hell


# Running the model


> To run on CPU in CUDA enabled machine, modify update `CUDA` variable in `recognizer/__init__.py`


Dockerfile is added to help with setting up environment. 

Run below command from the root directory to build the image
    ``` docker build -t number_recognizer:0.1 .```


Create container and get tty - ```docker run --name recognizer_app -it --entrypoint bash number_recognizer:0.1```

~~Run ```python recognizer/detect.py --model file.pt --image```~~

To setup in virtualenv
Create and active your env, then
- ```pip install -r requirements.txt``` 

- ```python setup.py install``` Make sure python is using local instance. More at https://stackoverflow.com/a/5979776/7445772


> **_Please note:_**
> - Evaluation/inference using python script - detect.py is not working, need more time to analyse the issue with packagings
>   - Inference on notebook is possible
> 