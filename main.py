import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import *

import re

import pickle
import argparse

#SKLEARN
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score

from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier



def read_data():
    data_train = pd.read_csv('../Datasets/drugsComTrain_raw.tsv', sep='\t')
    data_test = pd.read_csv('../Datasets/drugsComTest_raw.tsv', sep='\t')

    return data_train, data_test

data_train,data_test=read_data()


def df_train_test():
    train_df = data_train[['condition', 'review']]
    test_df = data_test[['condition', 'review']]
    train_df = train_df.dropna()
    test_df = test_df.dropna()

    def resub(review):
        review = re.sub(r"&#039;", "'", review)
        return review

    train_df.review = train_df.review.apply(resub)
    test_df.review = test_df.review.apply(resub)

    train_df = train_df[~train_df.condition.str.contains('</span>')]
    test_df = test_df[~test_df.condition.str.contains('</span>')]

    return train_df, test_df

train_df,test_df=df_train_test()


def reduce_conditions(value):
    cond = train_df.condition.value_counts() > value

    def g(condition):
        if cond[condition]:
            return condition
        else:
            return 'other'

    train_df['condcopy'] = train_df['condition'].apply(g)

    s = set(train_df['condcopy'])
    in_s = test_df['condition'].isin(s)
    test_df['condcopy'] = test_df['condition']
    test_df['condcopy'][~in_s] = 'other'

    len_train = len(set(train_df.condcopy))
    len_test = len(set(test_df.condcopy))

    other_train = train_df.condcopy.value_counts()['other'] / train_df.shape[0] * 100
    other_test = test_df.condcopy.value_counts()['other'] / test_df.shape[0] * 100
    print('Nr conditions Train: ', len_train, '\nNr conditions Test: ', len_test)
    print('Percentage "other", Train: ', other_train, '%')
    print('Percentate "other", Test: ', other_test, '%')

reduce_conditions(50)

ps = PorterStemmer()
def tknizer(text):
    words = re.sub(r"[^A-Za-z0-9\-]", " ", text).lower().split()
    words = [ps.stem(word) for word in words]
    return words

def cv(max_features=5000,stop_words=stopwords.words("english"),tokenizer=tknizer):
    try:
        cv_train=pickle.load(open('cv_train.pkl','rb'))
        cv_test=pickle.load(open('cv_test.pkl','rb'))
    except:
        cv=CountVectorizer(max_features=max_features,stop_words=stop_words,tokenizer=tokenizer)

        cv_train=cv.fit_transform(train_df.review)
        print('cv_train done.')
        cv_test=cv.transform(test_df.review)
        print('cv_test done.')

    return cv_train,cv_test

cv_train,cv_test=cv(10000)


def acc(model, X_train, X_test, y_train, y_test):
    preds_train = model.predict(X_train)
    acc_train = accuracy_score(preds_train, y_train)
    print('accuracy train done.')

    preds_test = model.predict(X_test)
    acc_test = accuracy_score(preds_test, y_test)
    print('accuracy test done.')

    print('Train error: ', acc_train, '\nTest error: ', acc_test)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Random Forests')
    parser.add_argument('--n_estimators', type=int)
    parser.add_argument('--max_depth', type=int)
    parser.add_argument('--min_samples_leaf', type=int)
    args = parser.parse_args()

    RFmodel = RandomForestClassifier(criterion="entropy",n_estimators=args.n_estimators, max_depth=args.max_depth, min_samples_leaf=args.min_samples_leaf,random_state=0, verbose=1, n_jobs=-1)
    print(RFmodel.get_params())

    RFmodel.fit(cv_train, train_df.condcopy)
    acc(RFmodel,cv_train,cv_test,train_df.condcopy,test_df.condcopy)