# Bounding Box Fusion
The repository is a part of Mitigating-Negative-Transfer
It contains the code for fusing detections of results obtained from different
mitigation methods. Experiments were carried out using various combinations of mitigated instances and model weights. 

# Description 

## Folders
|Name|Description|
|-----|---|
|Results| CSV files of Bounding Box coordinates and Confidence
|aicrowd_submissions| CSV files of Bounding Box coordinates in Competition Submission Format|

## Files

| Name | Description |
|------|--------------|
|fusion.ipynb|Code for Weighted Bounding Box Fusion|
|detections.py|Code for drawing bounding boxes on images and saving to disk|
|aicrowd_format.py|Code to convert predictions in Results folder to Competition accepted format|


