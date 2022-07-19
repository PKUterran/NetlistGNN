# NetlistGNN

## For simple test

Note that **NetlistGNN** is the former name of our model, now it's revised to **Circuit GNN**, but some parts of the code are not revised.

The download links of ISPD2011 and DAC2012 are listed in the paper.

Preprocessed by [DREAMPlace](https://github.com/limbo018/DREAMPlace.git)

As the raw data are too big, only `superblue19` is included here.

Run following command to train the **Circuit GNN** on `superblue19`:
```commandline
python script_train_sample.py
```

## For reproduction

1. Get and store the output data of DREAMPlace in `{data_dir}`.
2. Config and run `data/script_process.py` to get processed data (in `{data_dir}-processed`).
3. Config and run `script_train.py` to train **CircuitGNN**:
   1. Beyond the args, `train_dataset_names`, `validate_dataset_name` and `test_dataset_name` should also be configured.
   2. At the first time of training, a Circuit Graph will be generated in `{data_dir}-processed/hetero_{given_iter}`. This might cost about an hour.
