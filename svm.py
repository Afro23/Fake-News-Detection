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
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, f1_score, classification_report
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
import sklearn.metrics as metrics
import scikitplot as skplt

def build_confusion_matrix(classifier,lab):
    
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
    plt.title(lab)
    plt.tight_layout()
    plt.xlabel('Predicted label\nAccuracy = {}'.format(accuracy))
    plt.savefig('images/'+ lab +'.png')
    plt.show()

    return (print(lab),print('Total statements classified:', len(DataPrep.train_news)),
    print('Accuracy:', accuracy),
    print('score length', len(scores)),
    print('Confusion matrix:'),
    print(confusion))
    
#building Linear SVM classfier
svm_pipeline = Pipeline([
        ('svmCV',FeatureSelection.countV),
        ('svm_clf',svm.LinearSVC())
        ])

svm_pipeline.fit(DataPrep.train_news['Statement'],DataPrep.train_news['Label'])
predicted_svm = svm_pipeline.predict(DataPrep.test_news['Statement'])
np.mean(predicted_svm == DataPrep.test_news['Label'])
build_confusion_matrix(svm_pipeline,"Linear SVM")

#linear SVM classifier
svm_pipeline_ngram = Pipeline([
        ('svm_tfidf',FeatureSelection.tfidf_ngram),
        ('svm_clf',svm.LinearSVC())
        ])

svm_pipeline_ngram.fit(DataPrep.train_news['Statement'],DataPrep.train_news['Label'])
predicted_svm_ngram = svm_pipeline_ngram.predict(DataPrep.test_news['Statement'])
np.mean(predicted_svm_ngram == DataPrep.test_news['Label'])
build_confusion_matrix(svm_pipeline,"SVM N-Gram")


#Linear SVM 
parameters = {'svm_tfidf__ngram_range': [(1, 1), (1, 2),(1,3),(1,4),(1,5)],
               'svm_tfidf__use_idf': (True, False),
               'svm_tfidf__smooth_idf': (True, False),
               'svm_clf__penalty': ('l1','l2'),
}

gs_clf = GridSearchCV(svm_pipeline_ngram, parameters, n_jobs=-1)
gs_clf = gs_clf.fit(DataPrep.train_news['Statement'][:10000],DataPrep.train_news['Label'][:10000])

gs_clf.best_score_
gs_clf.best_params_
gs_clf.cv_results_
