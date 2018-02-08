import re
import string
import sklearn
import csv
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

## PARSING TRAINING DATA INTO A LIST OF COMMENTS AND SCORES - BENIGN/ATTACK
train_text=[]
train_score=[]
with open('train.tsv', mode='r', encoding = 'utf-8') as tsv_train:
    for line in tsv_train:
        line=line.split('\t')
        line[1] = re.sub('[' + string.punctuation + ']', '', line[1])
        train_score.append(line[0])
        train_text.append(line[1])

## PARSING TEST DATA INTO A LIST OF COMMENTS AND SCORES - BENIGN/ATTACK
test_score=[]
test_text=[]
with open('test.tsv', mode='r', encoding = 'utf=8') as tsv_test:
    for line in tsv_test:
        line=line.split('\t')
        line[1] = re.sub('[' + string.punctuation + ']', '', line[1])
        test_score.append(line[0])
        test_text.append(line[1])

## FIRST ROUND OF TRAINING - USING text_clf
text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', LinearSVC()),])
text_clf.fit(train_text, train_score)
predicted = text_clf.predict(test_text)

for i in [0.95,0.96,0.97,0.98,0.98,0.99,1,1.01,1.02,1.03]:
    pipeline = Pipeline([('tfidf', TfidfVectorizer(ngram_range=(1,3),max_df = 0.92,stop_words='english')),('clf', LinearSVC(C=i))])
    pipeline.fit(train_text, train_score)
    predicted = pipeline.predict(test_text)

## SECOND ROUND OF TRAINING - USING Random Forest Classifier
random_forest = Pipeline([('tfidf', TfidfVectorizer(ngram_range=(1,2),max_df = 0.9,stop_words='english')),('clf', RandomForestClassifier(n_estimators=12))])
random_forest.fit(train_text, train_score)
predicted = random_forest.predict(test_text)
test_score = test_score[1:]
## WRITING TO CSV FILE AS OUTPUT
with open('output.csv', mode='w', newline = '') as csvfile:
    spamwriter=csv.writer(csvfile,delimiter=',')
    spamwriter.writerow(['Id','Category'])
    for id, category in zip(test_score,predicted):
        spamwriter.writerow([id,category])
