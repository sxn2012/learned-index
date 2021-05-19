# Machine Learning Index Models
This is my final research project of *Master of Science (Computer Science)* in *the University of Melbourne*.

This repository is the implementation details of machine learning index models.

## Introduction

- We have reproduced B-Tree and deep learning indexing models mentioned in Kraska et al. (2018) "The case for learned index structures".
    - use a fully-connected neural network to build indexes
    - try to run deep learning models on GPUs and TPUs
- We have also used other learned models
    - Linear Regression
    - Naïve Bayes
    - Decision Trees
    - many others...

- We have conducted experiments in many types of datasets
    - build indexes on numbers
        - single-column numbers
        - multiple-column numbers
    - build indexes on strings
        - text strings: strings that can be split into words via spaces
        - non-text strings: strings that cannot be divided into words
- We have tuned hyper-parameters
    - record the results of each experiments
    - find out how these hyper-parameters affect each model's performance

## Implementation Details

### Environments

- Python 3.7+
- Scikit-learn 
- Tensorflow 2.0+
- Numpy
- Spacy
- chars2vec

Make sure you have installed these environments in your device. If you haven't install the packages, try running `pip install [package name]` to install them before you run the code.

If you want to run the learned indexing structures on string datasets, you should also execute the following command:

```shell
python3 -m spacy download en_core_web_md

pip install keras-word-char-embd

pip install chars2vec
```

### Datasets

The datasets in this research project are ...

- randomly generated
- from UCI Repository

The details can be seen in `dataset` folder.

Each dataset contains a training set ,`data.csv`, and a testing set, `data_test.csv`. Ensure you have downloaded both of the dataset files, and place both files in a separate folder. Record the directory of that folder before you run the code. 

### How to run the code

If you want to run the experiment on GPU/TPU, ensure you have installed the correct version of Tensorflow.



If you want to index on single-column numbers, run the following command:

`python3 .\code\learned_index_1d_number.py -d [dataset path] -c [counting number] -p [hyper-parameter] -m [model name]`

If you want to index on multiple-column numbers, run the following command:

`python3 .\code\learned_index_nd_number.py -d [dataset path] -c [counting number] -p [hyper-parameter] -m [model name]`

If you want to index on strings, run the following command:

`python3 .\code\learned_index_string.py -d [dataset path] -c [counting number] -p [hyper-parameter] -m [model name] -e [type]`

- Parameter Descriptions:

  - `-d`: the directory where `data.csv` and `data_test.csv` are stored
  - `-c`: how many times will the code executes
  - `-p`: the hyper-parameters of the model
  - `-m`: the model name
  - `-e`: the type of string datasets

- Parameter requirement

  - `-d`: must be a directory, not a file

  - `-c`: must be an integer, >=1

  - `-p` ,`-m`: see the table below

    | Description        | `-m` value | `-p` requirements |
    | ------------------ | ---------- | ----------------- |
    | B-Tree             | BT         | integer, >=2      |
    | Linear Regression  | LR         | N/A               |
    | Ridge Regression   | RR         | float             |
    | K-Nearest Neighbor | KNN        | integer, >=1      |
    | Naïve Bayes        | NB         | N/A               |
    | Decision Tree      | DT         | integer, >=2      |
    | Neural Networks    | NN         | N/A               |

  - `-e`: 1 for text strings; 2 for non-text strings

- Examples:

  `python .\code\learned_index_1d_number.py -d ".\dataset\single column number\random data\data_10k" -c 5 -m NN`

  `python3 .\code\learned_index_nd_number.py -d ".\dataset\multiple column number\bike sharing data" -c 5 -p 70 -m DT`

  `python3 .\code\learned_index_string.py -d ".\dataset\single column text\health news" -c 5 -p 3 -m BT -e 1`

## Other Contents

The [notebooks](https://github.com/sxn2012/learned-index/tree/notebooks/notebooks) are available via the link, which can be run on Google Colab directly!

### File Function Descriptions

- `code`: stores all code files in this research project
	- [random_generation.py](https://github.com/sxn2012/learned-index/blob/master/code/random_generation.py): used for generating random datasets
	- [test_generation.py](https://github.com/sxn2012/learned-index/blob/master/code/test_generation.py): a test code file to make sure we generate different datasets each time
	- [learned_index_string.py](https://github.com/sxn2012/learned-index/blob/master/code/learned_index_string.py): used for indexing experiments on 1-dimention strings 
	- [learned_index_1d_number.py](https://github.com/sxn2012/learned-index/blob/master/code/learned_index_1d_number.py): used for indexing experiments on 1-dimention numbers
	- [learned_index_nd_number.py](https://github.com/sxn2012/learned-index/blob/master/code/learned_index_nd_number.py): used for indexing experiments on multiple-dimention numbers
	- [learned_index_figures.py](https://github.com/sxn2012/learned-index/blob/master/code/learned_index_figures.py): used for generating charts of this research project
- `dataset`: stores all datasets that are used in this research project
	- `single column number`: datasets of single column numbers
	- `multiple column number`: datasets of multiple column numbers
	- `single column text`: datasets of text strings
	- `single column non-text`: datasets of non-text strings
- `img`: stores some of the figures from our experimental results

## Copyright

Copyright © 2021, [Xinnan SHEN](https://github.com/sxn2012), released under the [GPL-3.0 License](https://github.com/sxn2012/learned-index/blob/master/LICENSE).

If you found this repository useful for your research, we appreciate you cite it.