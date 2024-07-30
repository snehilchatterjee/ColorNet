# ColorNet
## Introduction:
ColorNet is a sequential image coloring model created for the AMD Pervasive AI Developer Contest.

## Dependencies:
Install required packages using:
```
pip install -r requirements.txt
```

## Training
The model was trained on the blackclover\colored subdirectory of the japanese manga dataset which can be obtained from kaggle. [Link to Dataset](https://www.kaggle.com/datasets/chandlertimm/unified/data)

The model can be trained by running:
```
python train.py --data DATASET_PATH

Other parameters:

--batch_size           default: 2
--size                 default: 256
--tempo_length         default: 5
--pretrain_epochs      default: 5
--train_epochs         default: 10
```

## Inference
If you just wish to try out the model, the trained weights can be downloaded here: [Download Weights](https://drive.google.com/file/d/1WdknbykO5RLg-ydciHyOU864vEJQDTJl/view?usp=sharing). Copy the downloaded file to the weights subdirectory.

After obtaining the weights(through training or from the link provided above), the inference app can be run using:
```
python app.py
```

A script to convert colored images to greyscale has also been included

Usage:
```
python convert.py (Path to directory contining colored images) (Path to store colored images)

Example:
python convert.py ./Dataset/ds/onepiece/colored ./Dataset/ds/onepiece/L
```




