# SSHNet: Unsupervised Cross-modal Homography Estimation via Problem Reformulation and Split Optimization

This is the implementation of the paper "SSHNet: Unsupervised Cross-modal Homography Estimation via Problem Reformulation and Split Optimization"

## Requirements

- Create a new anaconda environment and install all required packages before running the code.

```
conda create --name SSHNet python=3.9
conda activate SSHNet
pip install -r requirements.txt
```

## Train

```
# SSHNet
python -u train.py --gpuid 0 --dataset ggmap --note exp
# SSHNet-D
python -u train_distillation.py --gpuid 0 --dataset ggmap --checkpoint ./logs/optsar/model_iter_120000 --note exp
```

## Test

```
python -u train.py --gpuid 0 --mode test --dataset ggmap --checkpoint ./logs/optsar/model_iter_120000  --note test
```

## License

This project is released under the Apache 2.0 license.
