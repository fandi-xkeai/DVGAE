# Disentangled (Variational) Graph Auto-Encoder

## Requirements

- Pytorch 1.12.1
- Python 3.7
- DGL 1.1.2
- numpy 1.21.5

## Run the demo

Run with following (available dataset: "cora", "citeseer", "pubmed")

```
python train.py  --datname cora
```
```
python train_DGAE.py --datname cora
```

**Note**: If you want to train by dataset from website, you should download folder https://github.com/kimiyoung/planetoid/tree/master/data. Then put it under project folder.



