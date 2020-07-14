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
import xgboost as xgb


def read_data():
    data_train = pd.read_csv('drugsComTrain_raw.tsv', sep='\t')
    data_test = pd.read_csv('drugsComTest_raw.tsv', sep='\t')

    return data_train, data_test

def df_train_test(data_train,data_test):
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


def reduce_conditions(value,train_df,test_df):
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


def cv(max_features=5000,stop_words=stopwords.words('english')):
    try:
        cv_train=pickle.load(open('cv_train.pkl','rb'))
        print('cv_train loaded')
        y_train=pickle.load(open('y_train.pkl','rb'))
        print('y_train loaded\n')
        cv_test=pickle.load(open('cv_test.pkl','rb'))
        print('cv_test loaded')
        y_test=pickle.load(open('y_test.pkl','rb'))
        print('y_test loaded\n')
    except:
        data_train,data_test=read_data()
        train_df,test_df=df_train_test(data_train,data_test)
        reduce_conditions(20,train_df,test_df)
	
        y_train=train_df.condcopy
        y_test=test_df.condcopy

        ps = PorterStemmer()
        def tknizer(text):
            words = re.sub(r"[^A-Za-z0-9\-]", " ", text).lower().split()
            words = [ps.stem(word) for word in words]
            return words
        print('Initiating vectorizer...')
        cv=CountVectorizer(max_features=max_features,stop_words=stop_words,tokenizer=tknizer)
        print('vectorizer done\n')

        cv_train=cv.fit_transform(train_df.review)
        pickle.dump(cv_train,open('cv_train.pkl','wb'))
        print('cv_train saved')
        pickle.dump(y_train,open('y_train.pkl','wb'))
        print('y_train saved\n')
        cv_test=cv.transform(test_df.review)
        pickle.dump(cv_test,open('cv_test.pkl','wb'))
        print('cv_test saved')
        pickle.dump(y_test,open('y_test.pkl','wb'))

    return cv_train,cv_test,y_train,y_test

cv_train,cv_test,y_train,y_test=cv(5000)


def acc(model, X_train, X_test, y_train, y_test):
    preds_train = model.predict(X_train)
    acc_train = accuracy_score(preds_train, y_train)
    print('accuracy train done.')

    preds_test = model.predict(X_test)
    acc_test = accuracy_score(preds_test, y_test)
    print('accuracy test done.')

    print('Train error: ', acc_train, '\nTest error: ', acc_test)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Random Forests & XGBoost')
    parser.add_argument("--model",type=str)
    parser.add_argument('--n_estimators', type=int)
    parser.add_argument('--max_depth', type=int)
    parser.add_argument('--max_features',type=int)	
    parser.add_argument('--min_samples_split',type=int)

    args = parser.parse_args()
    if args.model=='rfc':
        print('Initiating Random Forest Classifier')
        RFmodel = RandomForestClassifier(criterion="gini",n_estimators=args.n_estimators, max_depth=args.max_depth,max_features=args.max_features,min_samples_split=args.min_samples_split,random_state=45, verbose=1, n_jobs=-1)
        print(RFmodel.get_params())

        RFmodel.fit(cv_train, y_train)
        acc(RFmodel,cv_train,cv_test,y_train,y_test)
    if args.model=='xgbc':
        print('Initiating XGBoost Classifier')
        params = {'n_estimators': args.n_estimators, 'max_depth': args.max_depth, 'learning_rate': 0.1, 'objective': 'multi:softmax','verbosity': 1, 'n_jobs': -1}
        model_xgb = xgb.XGBClassifier(**params)
        print(model_xgb.get_params())
        model_xgb.fit(cv_train, y_train)
        acc(model_xgb, cv_train, cv_test, y_train, y_test)
    if args.model not in ['rfc','xgbc']:
        print('Model not recognized.\nModels recognized are "rfc" for Random Forest Classifier and "xgbc" for XGBoost Classifier.')
