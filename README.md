# U-Net Model on Oxford Pets Dataset

### Introduction

This repository contains a deep learning project that employs 
the U-Net architecture to perform image segmentation on the Oxford Pets Dataset. 
The goal is to identify and segment pets in images. 
The U-Net model is trained for 15 epochs, featuring 5 down-sampling and 5 up-sampling layers. 
My implementation achieved a score of 89%.


### Features

U-Net architecture for image segmentation.
Pre-processing scripts for the Oxford Pets Dataset.
Trained for 15 epochs.
5 down-sampling (encoder) and 5 up-sampling (decoder) layers.
Evaluation metrics and visualizations.


### Requirements

```
numpy==1.25.2
pandas==2.0.3
matplotlib==3.7.2
seaborn==0.12.2
scikit-learn==1.3.0
scipy==1.11.2
torch==2.0.1
torchaudio==2.0.2
torchvision==0.15.2
plotly==5.16.1
```







    Valid accuracy on 14 = 0.890987634333904



    
![png](README_files/README_22_1.png)
    



    
![png](README_files/README_22_2.png)
    



    
![png](README_files/README_22_3.png)
    



    
![png](README_files/README_22_4.png)
    



    
![png](README_files/README_22_5.png)
    



    
![png](README_files/README_22_6.png)
    



    
![png](README_files/README_22_7.png)
    



    
![png](README_files/README_22_8.png)
    



    
![png](README_files/README_22_9.png)
    



    
![png](README_files/README_22_10.png)
    



    
![png](README_files/README_22_11.png)
    



    
![png](README_files/README_22_12.png)
    


    Reached 89% accuracy on validation set. Stopping training.
    Valid accuracy = 0.890987634333904

