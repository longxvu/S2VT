## How to run:
First you should structure the data like below (or place data wherever you want
and change the data path in code accordingly):
```shell
└── sv2t
    ├── data
        ├── YouTubeClips
        ├── YouTubeClips_features_resnet
        └── AllVideoDescriptions.txt
        └── train_split.txt
        └── ...
    ├── runs
        ├── last_resnet_300.pth
        ├── last_resnet_300.pth
    ├── train.py
    └── demo.py
    └── ...
```
Install necessary package requirements:
```shell
python >= 3.8
numpy
tqdm
opencv_python
torch
torchvision
```
Then run:
```python
python preprocess.py    # Create feature files, only need to run once
python train.py
# or
python demo.py
# evaluation on test_set
python eval.py
```

