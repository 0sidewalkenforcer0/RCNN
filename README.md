# TensorFlow Implementation of Recurrent Control Neural Network for the Mountain Car Problem
We implement the Recurrent Control Neural Network (RCNN) to solve the mountain car problem based on https://www.elen.ucl.ac.be/Proceedings/esann/esannpdf/es2007-54.pdf

## Requirements
The requirements of our code are given in requirements.txt
Make sure you have at least Python 3.7 with pip installed.
Navigate to the root folder of the repository, where the file requirements.txt is located, and install all dependencies in the current Python environment with
```
pip install -r requirements.txt
```

## Sample Data
Generate dataset of 200 episodes, split 7:1:2 into training, validation and test sets and save in `data/200/`.
```
cd src/data_preprocessing
python data_preprocessing.py
```

## Train Model
Train a rnn and a rcnn model consecutively. The trained models are saved as checkpoints in the `model/rnn/` and `model/rcnn`, respectively.
```
cd src
python train.py
```

## Evaluation
Evaluation methods for rnn and rcnn are located in `src/rnn/d_evaluate.py` and `src/rcnn/c_evaluate.py`, where 'd' and 'c' stand for 'dynamics' and 'control'.

## Others
Currently, you can find our best performance RNN and RCNN model with window size = 8 in `/models`. They are trained on the dataset in `/data/200`, sampled over 200 episodes.



