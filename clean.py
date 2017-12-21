from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
from sys import stdout, argv
from utils import contraction, correction, stop_words, stemmer, split_hashtag
import itertools
import matplotlib.pyplot as plt
import pandas
import re

file_name = argv[1]
test = False
if len(argv) == 3 and argv[2] == 'test':
    test = True


def clean(tweet_text):

    #ignore non ascii characters
    tweet_text = tweet_text.encode('ascii','ignore')
    tweet_text = tweet_text.lower()

    #remove html tags
    tweet_text = BeautifulSoup(tweet_text, 'html.parser').get_text()

    #remove @ references
    tweet_text = re.sub(r'@\w+', ' ', tweet_text)

    #remove 'RT' text
    tweet_text = re.sub(r'(^| )rt ', ' ', tweet_text)

    #remove links
    tweet_text = re.sub(r'https?:\/\/\S*', ' ', tweet_text)

    #Voteeeeeeeee -> Votee
    tweet_text = ''.join(''.join(s)[:2] for _,s in itertools.groupby(tweet_text))

    #split hashtags
    useless_hashtag = ['tcot', 'tlot', 'ucot', 'p2b', 'p2', 'ccot']
    split_tweet_text = ''
    for word in tweet_text.split():
        if word.startswith('#'):
            split_words = split_hashtag(word[1:]) if word[1:] not in useless_hashtag else ' '
            if split_words:
                split_tweet_text += (' ' + correction(word[1:]) + ' ' + split_words)
            else:
                split_tweet_text += (' ' + correction(word[1:]))
        else:
            #expand contractions
            if word in contraction:
                split_tweet_text += ' ' + contraction[word]
            else:
                split_tweet_text += ' ' + correction(word)
    tweet_text = split_tweet_text

    #remove special char (except #) and contraction for hash tag
    tweet_text = re.sub('[^0-9a-zA-Z]',' ',tweet_text)

    #tokenize
    words = word_tokenize(tweet_text)

    #remove stopwords
    words = [stemmer.stem(w) for w in words if not w in stop_words]

    #join the words
    tweet_text = " ".join(words)

    return tweet_text


def clean_tweets_to_file(tweets, output_file_name):

    if not test:
        #remove non tagged and class 2 data
        tweets = tweets.loc[tweets['Class'].isin([1,-1,0])]
    #convert tweets to unicode
    tweets['Annotated tweet'] = tweets['Annotated tweet'].astype(unicode).dropna()

    f = open('data/' + output_file_name + ("_test" if test else "") + '_cleaned.json', 'w+')
    f.write('[')
    length = float(tweets.shape[0])
    for index, row in tweets.iterrows():
        cleaned_tweet = clean(row[0])
        label = "\"" if test else ("\", \"label\": " + str(row[1]))
        f.write("{\"text\": \"" + str(cleaned_tweet) + label + "},")
        stdout.write("\b"*50 + "Writing to %s: %.2f %%" % (output_file_name, index/length * 100))
    f.write(']')
    print ""
    f.close()


print "\n\nCleaning data...\n"

for candidate in ['Obama', 'Romney']:
    tweets = pandas.read_excel(file_name, sheetname=candidate, parse_cols='D:E')
    clean_tweets_to_file(tweets, candidate.lower())
