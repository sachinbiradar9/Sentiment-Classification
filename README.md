# Twitter Sentiment Classification
Labelled data classifying sentiment of tweets as positive, negative, neutral and mixed class are provided for both the candidates separately. A learning model was created using this labelled training data to classify sentiment of any given tweet as positive, negative or neutral class. Various classifiers are used to create the model to classify tweets, their relative performance are discussed in detail. The performance of the model is evaluated by F1score and Accuracy of the positive and negative class. Please refer [report](report.pdf) for details.

## Installation
`pip install scikit-learn`  
`pip install pandas`  
`pip installl matplotlib`  
`pip install numpy`  
`pip install nltk`  
`pip install bs4`  

## Usage
To clean the tweets - (test is optional paramenter to clean test data)  
`python clean.py tweet_file test`

To train and classify the tweets - (test is optional parameter for testing on tweets)  
`python classify.py test`

## Credits
- Lei Zhang, Riddhiman Ghosh, Mohamed Dekhil, Meichun Hsu, and Bing Liu. 2011. [_Combining lexicon-based and learning-based methods for twitter sentiment analysis_](http://www.hpl.hp.com/techreports/2011/HPL-2011-89.html)
- Steven Bird, Ewan Klein, and Edward Loper. 2009. [_Natural Language Processing with Python, O’Reilly Media_](http://shop.oreilly.com/product/9780596516499.do)
