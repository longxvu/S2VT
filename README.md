## How to run:
First you should structure the data like below (or place data wherever you want
and change the data path in code accordingly):
```shell
└── sv2t
    ├── data
        ├── YouTubeClips
        └── AllVideoDescriptions.txt
    ├── train.py
    └── demo.py
    └── ...
```
Package requirements:
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
```

