# DVIP_ICCV1_implementation
This repository is for the course "Data Visualisation and Image Processing" course final project

"""
Names of the Team members are 
1. Syed, Nizam Uddin
2. Orilade, Obahi Augustine
3. Mathumo, Gaolefufa Kefilwe

""""

## What problem are we solving?
* Domain adaption Object Detection on KITTI dataset using YOLO and Faster R-CNN
* This repository is going to demonstrate object detection on KITTI dataset using three retrained object detectors: YOLOv2, YOLOv3, Faster R-CNN and compare their performance.


# Method _ 1


## Data Preperation
* We used KITTI object 2D (http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d) for training YOLO -> we call it as  "SOURCE DOMAIN" 
    and used KITTI raw data ( http://www.cvlibs.net/datasets/kitti/raw_data.php) for testig -> we named it as "TARGET DOMAIN"

## Convert KITTI lables
To simplify the labels, we combined 9 original KITTI labels into 6 classes:
* Car
* Van
* Truck
* Tram
* Pedestrian
* Cyclist

### Note: YOLO needs the bounding box format as (center_x, center_y, width, height), instead of using typical format for KITTI. hence we converted KITTI dataset into COCO format using the github repo (https://github.com/TillBeemelmanns/cityscapes-to-coco-conversion)


## YOLO configurations
YOLO source code is available here (https://github.com/yizhou-wang/darknet-kitti). To train YOLO, beside training data and labels, we need the following documents: kitti.data, kitti.names, and kitti-yolovX.cfg. The data and name files is used for feeding directories and variables to YOLO. The configuration files kittiX-yolovX.cfg for training on KITTI is located at

YOLOv2: /darknet/cfg/kitti6-yolov2.cfg (https://github.com/yizhou-wang/darknet-kitti/blob/master/cfg/kitti6-yolov2.cfg)
YOLOv3: /darknet/cfg/kitti6-yolov3.cfg (https://github.com/yizhou-wang/darknet-kitti/blob/master/cfg/kitti6-yolov3.cfg)

## Details of configurations
Open the configuration file yolovX-voc.cfg and change the following parameters:

[net]
#### Training
batch=64
subdivisions=8
height=370
width=1224

[region]
classes=6

random=0  # remove resizing step
#### last convolutional layer
[convolutional]
filters=55

### do the same thing for the 3 yolo layers
[convolutional]
filters=33



## Evaluation results
For object detection, people often use a metric called mean average precision (mAP) to evaluate the performance of a detection algorithm. mAP is defined as the average of the maximum precision at different recall values.  I also count the time consumption for each detection algorithms. Note that the KITTI evaluation tool only cares about object detectors for the classes Car, Pedestrian, and Cyclist but do not count Van, etc. as false positives for cars.

## Quantitative results for YOLOv2
The results of mAP for KITTI using original YOLOv2 with input resizing.

Benchmark    Easy    Moderate    Hard
Car    45.32%    28.42%    12.97%
Pedestrian    18.34%    13.90%    9.81%
Cyclist    8.71%    5.40%    3.02%


The results of mAP for KITTI using modified YOLOv2 without input resizing.

Benchmark    Easy    Moderate    Hard
Car    88.17%    78.70%    69.45%
Pedestrian    60.44%    43.69%    43.06%
Cyclist    55.00%    39.29%    32.58%


Quantitative results for YOLOv3
The results of mAP for KITTI using modified YOLOv3 without input resizing.

Benchmark    Easy    Moderate    Hard
Car    56.00%    36.23%    29.55%
Pedestrian    29.98%    22.84%    22.21%
Cyclist    9.09%    9.09%    9.09%

Quantitative results for Faster R-CNN
The results of mAP for KITTI using retrained Faster R-CNN.

Benchmark    Easy    Moderate    Hard
Car    84.81%    86.18%    78.03%
Pedestrian    76.52%    59.98%    51.84%
Cyclist    74.72%    56.83%    49.60%

## Execution time analysis
we also analyze the execution time for the three models. YOLOv2 and YOLOv3 are claimed as real-time detection models so that for KITTI, they can finish object detection less than 40 ms per image. While YOLOv3 is a little bit slower than YOLOv2. However, Faster R-CNN is much slower than YOLO.


Model    Inference Time (per frame)
YOLOv2    15 ms
YOLOv3    35 ms
Faster R-CNN    2763 ms

## Conclusion
We implemented three kinds of object detection models, i.e., YOLOv2, YOLOv3, and Faster R-CNN, on KITTI 2D object detection dataset.  During the implementation, we did the following:

* pre-processed data and labels
* retrained and modified the models
* inferred testing results using retrained models
* evaluated the detection performance

[**The above work was impossible without this github repo : https://github.com/yizhou-wang/darknet-kitti **]



# Method _ 2

### Object Detection using YOLO system using pre-trained model with darknet framework. [Credits: https://pjreddie.com/]
* The pre-trained models are trained on COCO dataset but we are testing these models with CITYSCAPE dataset and the results are with good accuracy.
* This technique useds completely different approach, applying a single neural network to the full image to predict the objects in the image. 
* This network divides the image into regions and predicts bounding boxes and probabilities for each region. These bounding boxes are weighted by the predicted probabilities.

### step1: Installing Darknet

git clone https://github.com/pjreddie/darknet
cd darknet
make

### Step2: Downloading pretrained weight

wget https://pjreddie.com/media/files/yolov3.weights

### Then run the detector

./darknet detect cfg/yolov3.cfg yolov3.weights data/foggy_image_traffic.jpg


### Output

layer     filters    size              input                output
    0 conv     32  3 x 3 / 1   608 x 608 x   3   ->   608 x 608 x  32  0.639 BFLOPs
    1 conv     64  3 x 3 / 2   608 x 608 x  32   ->   304 x 304 x  64  3.407 BFLOPs
    2 conv     32  1 x 1 / 1   304 x 304 x  64   ->   304 x 304 x  32  0.379 BFLOPs
    3 conv     64  3 x 3 / 1   304 x 304 x  32   ->   304 x 304 x  64  3.407 BFLOPs
    4 res    1                 304 x 304 x  64   ->   304 x 304 x  64
    5 conv    128  3 x 3 / 2   304 x 304 x  64   ->   152 x 152 x 128  3.407 BFLOPs
    6 conv     64  1 x 1 / 1   152 x 152 x 128   ->   152 x 152 x  64  0.379 BFLOPs
    7 conv    128  3 x 3 / 1   152 x 152 x  64   ->   152 x 152 x 128  3.407 BFLOPs
    8 res    5                 152 x 152 x 128   ->   152 x 152 x 128
    9 conv     64  1 x 1 / 1   152 x 152 x 128   ->   152 x 152 x  64  0.379 BFLOPs
   10 conv    128  3 x 3 / 1   152 x 152 x  64   ->   152 x 152 x 128  3.407 BFLOPs
   11 res    8                 152 x 152 x 128   ->   152 x 152 x 128
   12 conv    256  3 x 3 / 2   152 x 152 x 128   ->    76 x  76 x 256  3.407 BFLOPs
   13 conv    128  1 x 1 / 1    76 x  76 x 256   ->    76 x  76 x 128  0.379 BFLOPs
   14 conv    256  3 x 3 / 1    76 x  76 x 128   ->    76 x  76 x 256  3.407 BFLOPs
   15 res   12                  76 x  76 x 256   ->    76 x  76 x 256
   16 conv    128  1 x 1 / 1    76 x  76 x 256   ->    76 x  76 x 128  0.379 BFLOPs
   17 conv    256  3 x 3 / 1    76 x  76 x 128   ->    76 x  76 x 256  3.407 BFLOPs
   18 res   15                  76 x  76 x 256   ->    76 x  76 x 256
   19 conv    128  1 x 1 / 1    76 x  76 x 256   ->    76 x  76 x 128  0.379 BFLOPs
   20 conv    256  3 x 3 / 1    76 x  76 x 128   ->    76 x  76 x 256  3.407 BFLOPs
   21 res   18                  76 x  76 x 256   ->    76 x  76 x 256
   22 conv    128  1 x 1 / 1    76 x  76 x 256   ->    76 x  76 x 128  0.379 BFLOPs
   23 conv    256  3 x 3 / 1    76 x  76 x 128   ->    76 x  76 x 256  3.407 BFLOPs
   24 res   21                  76 x  76 x 256   ->    76 x  76 x 256
   25 conv    128  1 x 1 / 1    76 x  76 x 256   ->    76 x  76 x 128  0.379 BFLOPs
   26 conv    256  3 x 3 / 1    76 x  76 x 128   ->    76 x  76 x 256  3.407 BFLOPs
   27 res   24                  76 x  76 x 256   ->    76 x  76 x 256
   28 conv    128  1 x 1 / 1    76 x  76 x 256   ->    76 x  76 x 128  0.379 BFLOPs
   29 conv    256  3 x 3 / 1    76 x  76 x 128   ->    76 x  76 x 256  3.407 BFLOPs
   30 res   27                  76 x  76 x 256   ->    76 x  76 x 256
   31 conv    128  1 x 1 / 1    76 x  76 x 256   ->    76 x  76 x 128  0.379 BFLOPs
   32 conv    256  3 x 3 / 1    76 x  76 x 128   ->    76 x  76 x 256  3.407 BFLOPs
   33 res   30                  76 x  76 x 256   ->    76 x  76 x 256
   34 conv    128  1 x 1 / 1    76 x  76 x 256   ->    76 x  76 x 128  0.379 BFLOPs
   35 conv    256  3 x 3 / 1    76 x  76 x 128   ->    76 x  76 x 256  3.407 BFLOPs
   36 res   33                  76 x  76 x 256   ->    76 x  76 x 256
   37 conv    512  3 x 3 / 2    76 x  76 x 256   ->    38 x  38 x 512  3.407 BFLOPs
   38 conv    256  1 x 1 / 1    38 x  38 x 512   ->    38 x  38 x 256  0.379 BFLOPs
   39 conv    512  3 x 3 / 1    38 x  38 x 256   ->    38 x  38 x 512  3.407 BFLOPs
   40 res   37                  38 x  38 x 512   ->    38 x  38 x 512
   41 conv    256  1 x 1 / 1    38 x  38 x 512   ->    38 x  38 x 256  0.379 BFLOPs
   42 conv    512  3 x 3 / 1    38 x  38 x 256   ->    38 x  38 x 512  3.407 BFLOPs
   43 res   40                  38 x  38 x 512   ->    38 x  38 x 512
   44 conv    256  1 x 1 / 1    38 x  38 x 512   ->    38 x  38 x 256  0.379 BFLOPs
   45 conv    512  3 x 3 / 1    38 x  38 x 256   ->    38 x  38 x 512  3.407 BFLOPs
   46 res   43                  38 x  38 x 512   ->    38 x  38 x 512
   47 conv    256  1 x 1 / 1    38 x  38 x 512   ->    38 x  38 x 256  0.379 BFLOPs
   48 conv    512  3 x 3 / 1    38 x  38 x 256   ->    38 x  38 x 512  3.407 BFLOPs
   49 res   46                  38 x  38 x 512   ->    38 x  38 x 512
   50 conv    256  1 x 1 / 1    38 x  38 x 512   ->    38 x  38 x 256  0.379 BFLOPs
   51 conv    512  3 x 3 / 1    38 x  38 x 256   ->    38 x  38 x 512  3.407 BFLOPs
   52 res   49                  38 x  38 x 512   ->    38 x  38 x 512
   53 conv    256  1 x 1 / 1    38 x  38 x 512   ->    38 x  38 x 256  0.379 BFLOPs
   54 conv    512  3 x 3 / 1    38 x  38 x 256   ->    38 x  38 x 512  3.407 BFLOPs
   55 res   52                  38 x  38 x 512   ->    38 x  38 x 512
   56 conv    256  1 x 1 / 1    38 x  38 x 512   ->    38 x  38 x 256  0.379 BFLOPs
   57 conv    512  3 x 3 / 1    38 x  38 x 256   ->    38 x  38 x 512  3.407 BFLOPs
   58 res   55                  38 x  38 x 512   ->    38 x  38 x 512
   59 conv    256  1 x 1 / 1    38 x  38 x 512   ->    38 x  38 x 256  0.379 BFLOPs
   60 conv    512  3 x 3 / 1    38 x  38 x 256   ->    38 x  38 x 512  3.407 BFLOPs
   61 res   58                  38 x  38 x 512   ->    38 x  38 x 512
   62 conv   1024  3 x 3 / 2    38 x  38 x 512   ->    19 x  19 x1024  3.407 BFLOPs
   63 conv    512  1 x 1 / 1    19 x  19 x1024   ->    19 x  19 x 512  0.379 BFLOPs
   64 conv   1024  3 x 3 / 1    19 x  19 x 512   ->    19 x  19 x1024  3.407 BFLOPs
   65 res   62                  19 x  19 x1024   ->    19 x  19 x1024
   66 conv    512  1 x 1 / 1    19 x  19 x1024   ->    19 x  19 x 512  0.379 BFLOPs
   67 conv   1024  3 x 3 / 1    19 x  19 x 512   ->    19 x  19 x1024  3.407 BFLOPs
   68 res   65                  19 x  19 x1024   ->    19 x  19 x1024
   69 conv    512  1 x 1 / 1    19 x  19 x1024   ->    19 x  19 x 512  0.379 BFLOPs
   70 conv   1024  3 x 3 / 1    19 x  19 x 512   ->    19 x  19 x1024  3.407 BFLOPs
   71 res   68                  19 x  19 x1024   ->    19 x  19 x1024
   72 conv    512  1 x 1 / 1    19 x  19 x1024   ->    19 x  19 x 512  0.379 BFLOPs
   73 conv   1024  3 x 3 / 1    19 x  19 x 512   ->    19 x  19 x1024  3.407 BFLOPs
   74 res   71                  19 x  19 x1024   ->    19 x  19 x1024
   75 conv    512  1 x 1 / 1    19 x  19 x1024   ->    19 x  19 x 512  0.379 BFLOPs
   76 conv   1024  3 x 3 / 1    19 x  19 x 512   ->    19 x  19 x1024  3.407 BFLOPs
   77 conv    512  1 x 1 / 1    19 x  19 x1024   ->    19 x  19 x 512  0.379 BFLOPs
   78 conv   1024  3 x 3 / 1    19 x  19 x 512   ->    19 x  19 x1024  3.407 BFLOPs
   79 conv    512  1 x 1 / 1    19 x  19 x1024   ->    19 x  19 x 512  0.379 BFLOPs
   80 conv   1024  3 x 3 / 1    19 x  19 x 512   ->    19 x  19 x1024  3.407 BFLOPs
   81 conv    255  1 x 1 / 1    19 x  19 x1024   ->    19 x  19 x 255  0.189 BFLOPs
   82 yolo
   83 route  79
   84 conv    256  1 x 1 / 1    19 x  19 x 512   ->    19 x  19 x 256  0.095 BFLOPs
   85 upsample            2x    19 x  19 x 256   ->    38 x  38 x 256
   86 route  85 61
   87 conv    256  1 x 1 / 1    38 x  38 x 768   ->    38 x  38 x 256  0.568 BFLOPs
   88 conv    512  3 x 3 / 1    38 x  38 x 256   ->    38 x  38 x 512  3.407 BFLOPs
   89 conv    256  1 x 1 / 1    38 x  38 x 512   ->    38 x  38 x 256  0.379 BFLOPs
   90 conv    512  3 x 3 / 1    38 x  38 x 256   ->    38 x  38 x 512  3.407 BFLOPs
   91 conv    256  1 x 1 / 1    38 x  38 x 512   ->    38 x  38 x 256  0.379 BFLOPs
   92 conv    512  3 x 3 / 1    38 x  38 x 256   ->    38 x  38 x 512  3.407 BFLOPs
   93 conv    255  1 x 1 / 1    38 x  38 x 512   ->    38 x  38 x 255  0.377 BFLOPs
   94 yolo
   95 route  91
   96 conv    128  1 x 1 / 1    38 x  38 x 256   ->    38 x  38 x 128  0.095 BFLOPs
   97 upsample            2x    38 x  38 x 128   ->    76 x  76 x 128
   98 route  97 36
   99 conv    128  1 x 1 / 1    76 x  76 x 384   ->    76 x  76 x 128  0.568 BFLOPs
  100 conv    256  3 x 3 / 1    76 x  76 x 128   ->    76 x  76 x 256  3.407 BFLOPs
  101 conv    128  1 x 1 / 1    76 x  76 x 256   ->    76 x  76 x 128  0.379 BFLOPs
  102 conv    256  3 x 3 / 1    76 x  76 x 128   ->    76 x  76 x 256  3.407 BFLOPs
  103 conv    128  1 x 1 / 1    76 x  76 x 256   ->    76 x  76 x 128  0.379 BFLOPs
  104 conv    256  3 x 3 / 1    76 x  76 x 128   ->    76 x  76 x 256  3.407 BFLOPs
  105 conv    255  1 x 1 / 1    76 x  76 x 256   ->    76 x  76 x 255  0.754 BFLOPs
  106 yolo
  
  
  
### Loading weights from yolov3.weights...Done!
* data/foggy_image_traffic.jpg: Predicted in 7.476640 seconds.
* truck: 96%
* car: 99%
* car: 98%
* car: 96%
* car: 94%
* car: 78%
* car: 100%
* car: 66%

