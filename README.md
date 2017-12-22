# Twitter Sentiment Classification
Labelled data classifying sentiment of tweets as positive, negative, neutral and mixed class are provided for both the candidates separately. A learning model was created using this labelled training data to classify sentiment of any given tweet as positive, negative or neutral class. Various classifiers are used to create the model to classify tweets, their relative performance are discussed in detail. The performance of the model is evaluated by F1score and Accuracy of the positive and negative class. Please refer [report](report.pdf) for details.

## Uasge
To clean the tweets - (test is optional paramenter to clean test data)

`python clean.py tweet_file test`

To train and classify the tweets - (test is optional parameter for testing on tweets)

`python classify.py test`
