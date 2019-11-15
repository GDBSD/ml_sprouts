# ML Sprouts

## Table of Contents
1. [Getting Started](#overview)
2. [Charting Basics](#charting)
3. [Comprehensive Overview of Traditional ML](#ml_foundation)
4. [Visualizing Data Science Projects with Streamlit](#streamlit)
5. [NLP Projects](#nlp)
7. [Getting More Sample Datasets](#tensorflow_ds)
7. [Miscellaneous Interesting Exercises](#misc)

<a name="overview"></a>
## Getting Started

ML Sprouts is intended to provide machine learning "sprouts", i.e. a project that contains "spouts"
yuu can transplant into your ML "garden" to "grow" a new project. Its secondary objective is to
provide a comprehensive survey of traditional machine learning methods, new deep learning tools, and
a collection of mostly toy examples using small datasets to speed things up examples and Jupyter notebooks.

#### IMPORTANT:
Most of the examples are very simple examples of the core concept. Read the docs to understand how to
make use of the various packages that are imported into a module.

Most of survey of traditional machine learning methods is extracted from the book _"Building Machine Learning
and Deep Learning Models on Google Cloud Platform"_. Note that, in general, training deep learning models works
best on a GPU instance. There are multiple alternatives for setting up a remote GPU. Some are free, some you
will have to pay for, usually by the minute. Prices on services like [Paperspace](https://www.paperspace.com/)
are quite reasonable.

NOTE: The O'Reilly link (below), and links in the code, point to pages in an e-book on their site. The book is
also available on [Amazon](https://www.amazon.com/Building-Machine-Learning-Models-Platform/dp/1484244699/ref=sr_1_1?keywords=Building+Machine+Learning+and+Deep+Learning+Models+on+Google+Cloud+Platform&qid=1573062357&sr=8-1)

Contributing: Please create a feature branch before committing code and do a PR when you're ready to
merge into the '''develop``` branch. Delete your feature branch after the merge.

Please comment you code and, if you're writing a Python script, add docstrings.

I use Anaconda which will probably load a bunch of packages you won't need but I'm lazy. All the the packages
are listed in requirements.txt. If you find you are missing a package that's essential to running your code
please add it to requirements.txt. I strongly recommend you create a virtual environment for any project you're
working on, even this one. My virtual environment (so I don't forget) is ```ml```

<a name="charting"></a>
## Charting Basics

<a name="ml_foundation"></a>
## Comprehensive Overview of Traditional ML
[Building Machine Learning and Deep Learning Models on Google Cloud Platform](https://learning.oreilly.com/library/view/building-machine-learning/9781484244708/html/Part_3.xhtml)

<a name="streamlit"></a>
## Visualizing Data Science Projects with Streamlit
[Seaborn documentation](https://seaborn.pydata.org/tutorial.html)

<a name="nlp"></a>
## NLP Projects
- ```sentiment_analysis.py``` and ```notebooks/iMDB_NLP.ipynb```:
-- [iMDB Sentiment Analysis](https://developers.google.com/machine-learning/guides/text-classification)
-- [Amazon demo with HuggingFace Transformers](https://github.com/microsoft/bert-stack-overflow/blob/master/1-Training/AzureServiceClassifier_Training.ipynb)
-- [BERT demo notebook](https://github.com/google-research/bert/blob/master/predicting_movie_reviews_with_bert_on_tf_hub.ipynb)

<a name="tensorflow_ds"></a>
## Getting More Sample Datasets

<a name="misc"></a>
## Miscellaneous Interesting Exercises

### Evolution of a salesman: A complete genetic algorithm tutorial for Python
From the [Medium blog post](https://towardsdatascience.com/evolution-of-a-salesman-a-complete-genetic-algorithm-tutorial-for-python-6fe5d2b3ca35) of the same name.