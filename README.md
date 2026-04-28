## Skeleton-Guided Deformable Alignment for Structurally Robust Few-Shot Font Generation

## Dependencies

### Prerequisites 

```
pytorch (>=1.7)
torchvision
dominate
visdom
tqdm
numpy
opencv-python  
scipy
sklearn
matplotlib  
pillow  
tensorboardX
scikit-image
scikit-learn
CUDA 12.4
```

## Dataset

[方正字库] [汉仪字体] [蒙纳字体] [新蒂字体] [造字工房]



# How to run

1. prepare dataset

   Put your font files to a folder and character file to charset

```
├──data_examples
│   └── train
│       ├── ContentImage
│       │   ├── char1.png
│       │   ├── char2.png
│       │   ├── char3.png
│       │   └── ...
│       └── TargetImage.png
│           ├── style1
│           │     ├──style1+char1.png
│           │     ├──style1+char2.png
│           │     └── ...
│           ├── style2
│           │     ├──style2+char1.png
│           │     ├──style2+char2.png
│           │     └── ...
│           ├── style3
│           │     ├──style3+char1.png
│           │     ├──style3+char2.png
│           │     └── ...
│           └── ...
```

2. Train 

   ```
   python train.py 
   ```

   

3.  Test 

```
  python test.py
```

