# Face Detector
Two custom PyTorch models for detecting faces in images. One architecture is a copy of MobileNetV2 and one is an architecture based on early versions of YOLO. Both model architectures only detect one face per image. The MobileNetV2 architecture is deployed to a Jetson Nano.

Created for CSCE 790 (Edge and Neuromorphic Computing) Spring 2025 at the University of South Carolina.

See the report at `docs/report.pdf`.

## Development
Most of the development was done on Kaggle, with the most recent notebook being [here](https://www.kaggle.com/code/reganwillis/fresh).

* Ran all hyperparameters optimization over learning rate and batch size for two architectures [here](https://www.kaggle.com/code/reganwillis/fresh/output?scriptVersionId=233691872&select=od_out_yolo_17445747593731036.csv).
* Picked two models from hyperparameter optimization to continue with [here](https://www.kaggle.com/code/reganwillis/fresh/output?scriptVersionId=233714662&select=hyperopt_out.csv).

## Submission
[Here](https://www.kaggle.com/competitions/assignment-2-1-real-time-face-detection/overview) is the link to the competition submission.

## Dataset
```
#!/bin/bash
curl -L -o ~/Downloads/dataset-face-detection-for-edge-computing-class.zip\
  https://www.kaggle.com/api/v1/datasets/download/icaslab/dataset-face-detection-for-edge-computing-class
```
