# Machine Learning Index Models
This is my final research project of *Master of Science (Computer Science)* in *the University of Melbourne*.



## Introduction

### Indexing Structures

- Traditionally, we use B-Trees to construct indexes in databases
    - B-Tree is fast and effective
    - B-Tree will always give us the correct position of the keys
-  Kraska et al. (2018)  have used deep learning as indexing structures
    - They have used a hierarchical neural network structure
    - Deep learning has potential to outperform traditional B-Tree indexing models

### Related Works

So far, the researchers ...

- only focused on the deep learning models
- have not tried to run such models on different environments, like
    - CPU
    - GPUs
    - TPUs
- have only done experiments on single-column numbers, like
    - ID numbers
    - Timestamps



## Research Contents

### What I have done

- reproduce B-Tree and deep learning indexing models mentioned in \cite{kraska2018case}
    - use a fully-connected neural network to build indexes
    - try to run deep learning models on GPUs and TPUs
- try other learned models
    - Linear Regression
    - Naïve Bayes
    - Decision Trees
    - many others...

- conduct experiments in many types of datasets
    - build indexes on numbers
        - single-column numbers
        - multiple-column numbers
    - build indexes on strings
        - text strings: strings that can be split into words via spaces
        - non-text strings: strings that cannot be divided into words
- tune hyper-parameters
    - record the results of each experiments
    - find out how these hyper-parameters affect each model's performance

### Evaluation Metrics

- Our focus is timings !
    - Index creation time, T1
    - Keys lookup time, T2
    - Error correction time, T3
- Compare Timings among...
    - different models
    - different hyper-parameters for each model

### Experiment processes

1. pre-process data
    - sort data in advance
    - split 1/3 of dataset as testing set, the entire dataset as the training set
    - for indexings on strings, map strings into vectors by using embeddings
2. build indexes
    - for traditional B-Trees, insert keys into the tree
    - for learned models, train the model with entire dataset
3. look up for keys in testing dataset
    - find keys in B-Trees, or
    - generate predictions of learned models
4. correct errors
    - only for learned models
    - using binary search



## Implementation Details

### Environments

- Google Colab (with GPU/TPU) enabled
- Python 3.7
- Scikit-learn
- Tensorflow

### Datasets

The datasets in this research project are ...

- randomly generated
- from UCI Repository

The details can be seen in `dataset` folder.

### How to run the code

First, make sure you have GPU/TPU enabled on Google Colab;

Then, upload the code in `code` folder to Google Colab;

You can run the code on server now!

### File Function Descriptions

- `code`: stores all code files in this research project
	- [random_generation.py](https://github.com/sxn2012/learned-index/blob/master/code/random_generation.py): used for generating random datasets
	- [test_generation.py](https://github.com/sxn2012/learned-index/blob/master/code/test_generation.py): a test code file to make sure we generate different datasets each time
	- [learned_index_1d_string.py](https://github.com/sxn2012/learned-index/blob/master/code/learned_index_1d_string.py): used for indexing experiments on 1-dimention strings 
	- [learned_index_1d_number.py](https://github.com/sxn2012/learned-index/blob/master/code/learned_index_1d_number.py): used for indexing experiments on 1-dimention numbers
	- [learned_index_multiple_number.py](https://github.com/sxn2012/learned-index/blob/master/code/learned_index_multiple_number.py): used for indexing experiments on multiple-dimention numbers
	- [learned_index_figures.py](https://github.com/sxn2012/learned-index/blob/master/code/learned_index_figures.py): used for generating charts of this research project
- `dataset`: stores all datasets that are used in this research project
	- `single column number`: datasets of single column numbers
	- `multiple column number`: datasets of multiple column numbers
	- `single column text`: datasets of text strings
	- `single column non-text`: datasets of non-text strings

## Findings

- Learned indexing structures are useful in almost all kinds of datasets, including ...
    - numbers
    - strings
- Traditional learned models are superior to deep learning and B-Tree structures, such as ...
    - Ridge Regression
    - Naïve Bayes
- Hyper-parameters affect some learned indexing models
- Deep learning may perform better with the help of GPUs or TPUs

##　References

Engineering, I. (2019). Chars2vec: character-based language model for handling real world texts.

Honnibal, M., Montani, I., Van Landeghem, S., and Boyd, A. (2020). spaCy: Industrial-strength Natural Language Processing in Python.

Kraska, T., Beutel, A., Chi, E. H., Dean, J., and Polyzotis, N. (2018). The case for learned index structures. In Proceedings of the 2018 International Conference on Management of Data, pages 489–504.

## Copyright

Copyright © 2021, [Xinnan SHEN](https://github.com/sxn2012), released under the [GPL-3.0 License](https://github.com/sxn2012/learned-index/blob/master/LICENSE).