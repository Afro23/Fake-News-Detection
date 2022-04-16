import DataPrep
import FeatureSelection
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, f1_score, classification_report
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
import sklearn.metrics as metrics
import scikitplot as skplt

def build_confusion_matrix(classifier,lab1,lab2):
    
    k_fold = KFold(n_splits=5)
    scores = []
    confusion = np.array([[0,0],[0,0]])

    for train_ind, test_ind in k_fold.split(DataPrep.train_news):
        train_text = DataPrep.train_news.iloc[train_ind]['Statement'] 
        train_y = DataPrep.train_news.iloc[train_ind]['Label']
    
        test_text = DataPrep.train_news.iloc[test_ind]['Statement']
        test_y = DataPrep.train_news.iloc[test_ind]['Label']
        
        classifier.fit(train_text,train_y)
        predictions = classifier.predict(test_text)
        
        confusion += confusion_matrix(test_y,predictions)
        score = f1_score(test_y,predictions)
        scores.append(score)
    accuracy = sum(scores)/len(scores)*100
    
    '''Plotting confusion matrix'''
    skplt.metrics.plot_confusion_matrix(test_y,predictions)
    plt.title(lab1+" "+lab2)
    plt.tight_layout()
    plt.xlabel('Predicted label\nAccuracy = {}'.format(accuracy))
    plt.savefig('images/'+ lab1 +"/"+lab2+'.png')
    plt.show()

    return (print(lab1+" "+lab2),print('Total statements classified:', len(DataPrep.train_news)),
    print('Accuracy:', accuracy),
    print('score length', len(scores)),
    print('Confusion matrix:'),
    print(confusion))

#using SVM Stochastic Gradient Descent on hinge loss
sgd_pipeline = Pipeline([
        ('svm2CV',FeatureSelection.countV),
        ('svm2_clf',SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3))
        ])

sgd_pipeline.fit(DataPrep.train_news['Statement'],DataPrep.train_news['Label'])
predicted_sgd = sgd_pipeline.predict(DataPrep.test_news['Statement'])
np.mean(predicted_sgd == DataPrep.test_news['Label'])
build_confusion_matrix(sgd_pipeline,"SGD","Hinge Loss")

#sgd classifier
sgd_pipeline_ngram = Pipeline([
         ('sgd_tfidf',FeatureSelection.tfidf_ngram),
         ('sgd_clf',SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3))
         ])

sgd_pipeline_ngram.fit(DataPrep.train_news['Statement'],DataPrep.train_news['Label'])
predicted_sgd_ngram = sgd_pipeline_ngram.predict(DataPrep.test_news['Statement'])
np.mean(predicted_sgd_ngram == DataPrep.test_news['Label'])
build_confusion_matrix(sgd_pipeline_ngram,"SGD","N-Gram")


