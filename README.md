1. DVC:

* `dvc init`
* `dvc add .\data\FashionMNIST\raw`
* `dvc add .\checkpoints`
* `dvc remote add -d storage ssh://izhanvarsky@109.188.135.85/storage/izhanvarsky/big_data_hw1_dvc`
* `dvc remote modify storage ask_password true`
* `dvc push`
* To fetch data from dvc use `dvc pull`

