# Env

```mamba env create -n FATS -f environment.yml ```

Refer to: https://github.com/conda-forge/miniforge#mambaforge，

# Dataset

- The script will automatically download the datasets the first time it runs `sh run_FeMNIST.sh` (except for Shakespeare). The initial download might be slow and exceed the client's waiting time for the server to start, causing the client to fail. If this happens, don't exit immediately; just wait for the download to finish. From the second run onward, the code will use the previously downloaded datasets.

- Shakespeare: Refer to https://github.com/TalwalkarLab/leaf.
After downloading and setting up the environment, generate the Shakespeare dataset based on the number of clients and the minimum number of samples per client specified in the paper.
    ```./preprocess.sh -s niid --sf 1.0 -k 100 -t sample```
Update the corresponding code in dataset.py to set the dataset location to the appropriate path on your computer.

# Run 

```sh run_FeMNIST.sh```

# Citation
If you find FATS useful for your research, please consider citing this paper:
```
@article{DBLP:journals/pvldb/TaoWPYCW24,
  author       = {Youming Tao and
                  Cheng{-}Long Wang and
                  Miao Pan and
                  Dongxiao Yu and
                  Xiuzhen Cheng and
                  Di Wang},
  title        = {Communication Efficient and Provable Federated Unlearning},
  journal      = {Proc. {VLDB} Endow.},
  volume       = {17},
  number       = {5},
  pages        = {1119--1131},
  year         = {2024},
  url          = {https://www.vldb.org/pvldb/vol17/p1119-wang.pdf},
  timestamp    = {Tue, 26 Mar 2024 22:14:30 +0100},
  biburl       = {https://dblp.org/rec/journals/pvldb/TaoWPYCW24.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```