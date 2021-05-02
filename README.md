# basic-pytorch

## Installizaion

- Install required packages :

```bash
pip3 install -r requirements.txt
```

- When to use anaconda virtual environment :

```bash
conda env create -f environment.yaml
conda activate basic_pytorch
```

## Usage

```
usage: main.py [-h] [--dataset DATASET] [--batch-size N] [--test-batch-size N] [--model MODEL] [--lr LR] [--momentum M] [--weight-decay W] [--epochs N] [--log-interval N] [--evaluate] [--pretrained]

Basic Pytorch

optional arguments:
  -h, --help            show this help message and exit
  --dataset DATASET     dataset: MNIST |
  --batch-size N        input batch size for training (default: 128)
  --test-batch-size N   input batch size for testing (default: 128)
  --model MODEL         dataset: LeNet5 |
  --lr LR               learning rate (default: 0.1)
  --momentum M          SGD momentum (default: 0.9)
  --weight-decay W, --wd W
                        weight decay (default: 5e-4)
  --epochs N            number of epochs to train (default: 200)
  --log-interval N      how many batches to wait before logging training status
  --evaluate            evaluate model
  --pretrained          pretrained
```

- Train new model
```bash
python main.py
```

- Evaluate pretrained model
```
python main.py --evaluate --pretrained
```
