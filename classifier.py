from imblearn.over_sampling import SMOTE
from sklearn import naive_bayes, svm, tree, ensemble, linear_model
from sklearn.feature_extraction import text
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import cross_val_predict
from sys import argv
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

test = False
if len(argv) == 2 and argv[1] == 'test':
    test = True

models = {
    'Multinomial NB': naive_bayes.MultinomialNB(),
    'Decision Tree': tree.DecisionTreeClassifier(random_state=0),
    'Random Forest': ensemble.RandomForestClassifier(criterion='entropy', n_jobs = 10),
    'Logistic Regression': linear_model.LogisticRegression(),
}


def vectorizer(train_tweets, test_tweets=[]):
    vectorizer = text.TfidfVectorizer(
        min_df = 0.00125,
        max_df = 0.7,
        sublinear_tf=True,
        use_idf=True,
        analyzer='word',
        ngram_range=(1,5)
    )
    train_vectors = vectorizer.fit_transform(train_tweets)
    test_vectors = vectorizer.transform(test_tweets) if test_tweets else []
    return train_vectors, test_vectors


def feature_generation(train_file, test_file=''):
    with open(train_file,"r") as train_data_text:
        train_data = json.load(train_data_text)
    train_tweets = []
    train_class = []
    for tweet in train_data:
        train_tweets.append(tweet['text'])
        train_class.append(tweet['label'])

    test_tweets = []
    if test_file:
        with open(test_file,"r") as test_data_text:
            test_data = json.load(test_data_text)
        for tweet in test_data:
            test_tweets.append(tweet['text'])

    train_vectors, test_vectors = vectorizer(train_tweets, test_tweets)
    return train_vectors, train_class, test_vectors


def plot_distribution(class_array, title):
    plt.figure(title)
    pd.DataFrame(class_array, columns = ['Class']).Class.value_counts().plot(
        kind='pie',
        autopct='%.2f %%',
    )
    plt.axis('equal')
    plt.title(title)


def over_sample(train_vectors, train_class):
    train_vectors = train_vectors.toarray()
    sm = SMOTE(random_state=42)
    train_vectors, train_class = sm.fit_sample(train_vectors, train_class)

    plot_distribution(train_class, 'After sampling')
    return train_vectors, train_class


def classify(classifier, train_vectors, train_class, test_vectors):
    if test:
        classifier.fit(train_vectors, train_class)
        preds = classifier.predict(test_vectors)
        return preds
    else:
        preds = cross_val_predict(classifier, train_vectors, train_class, cv=10)
        accScore = accuracy_score(train_class,preds)
        labels = [1,-1]
        precision = precision_score(train_class, preds, average=None,labels=labels)
        recall = recall_score(train_class,preds,average=None,labels=labels)
        f1score = f1_score(train_class,preds,average=None,labels=labels)
        return accScore, precision, recall, f1score


def train_classify(train_file, test_file):
    train_vectors, train_class, test_vectors = feature_generation(train_file, test_file)
    plot_distribution(train_class, train_file + ' Before sampling')
    train_vectors, train_class = over_sample(train_vectors, train_class)

    if test:
        eclf = ensemble.VotingClassifier(estimators=[
            ('nbm', models['Multinomial NB']),
            ('tree', models['Decision Tree']),
            ('rf', models['Random Forest']),
            ('lr', models['Logistic Regression']),
        ], voting='soft')
        preds = classify(eclf, train_vectors, train_class, test_vectors)
        f = open('data/' + candidate + '_predictions'  + '.txt', 'w+')
        for index, pred in enumerate(preds):
            f.write(str(index+1)+';;'+str(preds[index])+'\n')
        f.close()
    else:
        metrics = []
        for index, model in enumerate(models):
            print "Classifying using", model
            accScore, precision, recall, f1score = classify(models[model], train_vectors, train_class, test_vectors)
            metrics.append({})
            metrics[index]['Classifier'] = model
            metrics[index]['accuracy'] = accScore
            metrics[index]['possitive f1score'] = f1score[0]
            metrics[index]['negative f1score'] = f1score[1]
        pd.io.json.json_normalize(metrics).plot(kind='bar', x='Classifier')
        plt.title(train_file)
        plt.grid(True, axis='y')
        plt.ylim(ymax=1)
        plt.xticks(rotation=0)


for candidate in ['obama', 'romney']:
    print 'Running for', candidate, '...'
    test_file = 'data/' + candidate + '_test_cleaned.json' if test else ''
    train_classify('data/' + candidate + '_cleaned.json', test_file)

plt.show()
