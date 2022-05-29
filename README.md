# 2022 AI CUP: Orchid Species Recognition

https://tbrain.trendmicro.com.tw/Competitions/Details/20

## Problems

In `main.py`, we use self-defined dataset class to construct our dataset, and use `random_split` to split our dataset into training dataset and validation dataset.

Reference: https://stackoverflow.com/questions/65231299/load-csv-and-image-dataset-in-pytorch

However, <font color=#FF0000>**all of the dataset apply `train_tfm` transform method, which is NOT CORRECT in validation set.**</font> This is a critical dataset problem.

## Version 1: Use GoogleNet

### Result

The result of splitting ratio = 0.7, i.e., 7 pictures for training and 3 pictures for validation, is shown below.

```
[ Train | 073/10000 ] loss = 0.44498, acc = 0.87742
[ Valid | 073/10000 ] loss = 2.23616, acc = 0.54436
Saving model with validation loss 2.23616 and accuracy 0.54436
```

![ver.1_0.7](./img/ver.1_ratio_0.7.png)



## Version 2: ResNeXt50 32x4d

The result of splitting ratio = 0.7, i.e., 7 pictures for training and 3 pictures for validation, is shown below.

```
[ Train | 140/10000 ] loss = 0.48077, acc = 0.86562
[ Valid | 140/10000 ] loss = 2.34645, acc = 0.51072
Saving model with validation loss 2.34645 and accuracy 0.51072
```

```
[ Train | 388/10000 ] loss = 0.24341, acc = 0.93150
[ Valid | 388/10000 ] loss = 2.95899, acc = 0.55576
```

![ver.2_0.7](./img/ver.2_ratio_0.7.png)
