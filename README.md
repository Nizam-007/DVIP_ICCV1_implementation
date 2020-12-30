# DVIP_ICCV1_implementation
This repository is for the course "Data Visualisation and Image Processing" final project

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
## Faster RCNN model trained on Pascal VOC dataset with ResNet-50/SSD/YOLO.

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

```python

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
```

## Evaluation results
For object detection, people often use a metric called mean average precision (mAP) to evaluate the performance of a detection algorithm. mAP is defined as the average of the maximum precision at different recall values.  I also count the time consumption for each detection algorithms. Note that the KITTI evaluation tool only cares about object detectors for the classes Car, Pedestrian, and Cyclist but do not count Van, etc. as false positives for cars.

## Quantitative results for YOLOv2
The results of mAP for KITTI using original YOLOv2 with input resizing.

| Benchmark  | Easy   | Moderate | Hard   |
|------------|--------|----------|--------|
| Car        | 45.32% | 28.42%   | 12.97% |
| Pedestrian | 18.34% | 13.90%   | 9.81%  |
| Cyclist    | 8.71%  | 5.40%    | 3.02%  |


The results of mAP for KITTI using modified YOLOv2 without input resizing.

| Benchmark  | Easy   | Moderate | Hard   |
|------------|--------|----------|--------|
| Car        | 88.17% | 78.70%   | 69.45% |
| Pedestrian | 60.44% | 43.69%   | 43.06% |
| Cyclist    | 55.00% | 39.29%   | 32.58% |


Quantitative results for YOLOv3
The results of mAP for KITTI using modified YOLOv3 without input resizing.

| Benchmark  | Easy   | Moderate | Hard   |
|------------|--------|----------|--------|
| Car        | 56.00% | 36.23%   | 29.55% |
| Pedestrian | 29.98% | 22.84%   | 22.21% |
| Cyclist    | 9.09%  | 9.09%    | 9.09%  |

Quantitative results for Faster R-CNN
The results of mAP for KITTI using retrained Faster R-CNN.

| Benchmark  | Easy   | Moderate | Hard   |
|------------|--------|----------|--------|
| Car        | 84.81% | 86.18%   | 78.03% |
| Pedestrian | 76.52% | 59.98%   | 51.84% |
| Cyclist    | 74.72% | 56.83%   | 49.60% |

## Execution time analysis
we also analyze the execution time for the three models. YOLOv2 and YOLOv3 are claimed as real-time detection models so that for KITTI, they can finish object detection less than 40 ms per image. While YOLOv3 is a little bit slower than YOLOv2. However, Faster R-CNN is much slower than YOLO.


| Model        | Inference Time (per frame) |
|--------------|----------------------------|
| YOLOv2       | 15 ms                      |
| YOLOv3       | 35 ms                      |
| Faster R-CNN | 2763 ms                    |

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


```python

git clone https://github.com/pjreddie/darknet
cd darknet
make

```

### Step2: Downloading pretrained weight

```python
wget https://pjreddie.com/media/files/yolov3.weights
```
### Then run the detector
```python
./darknet detect cfg/yolov3.cfg yolov3.weights data/foggy_image_traffic.jpg
```

### Output
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

Input_Foggy_image: 
![alt text][input]

[input]: https://github.com/Nizam-007/DVIP_ICCV1_implementation/blob/main/model_predicted_images/input_foggy_traffic.jpg "Input Image"

Predicted_output_foggy_image: 
![alt text][output]

[output]: https://github.com/Nizam-007/DVIP_ICCV1_implementation/blob/main/model_predicted_images/predected_output_foggyTraffic.jpg "Predicted output image"


 
![alt text][sample]
[sample]: https://github.com/Nizam-007/DVIP_ICCV1_implementation/blob/main/model_predicted_images/Predicted_Images.png "sample output images"
