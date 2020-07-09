import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import *

import re
import argparse
import pickle
from sklearn.feature_extraction.text import CountVectorizer


def read_data():
    data_train = pd.read_csv('drugsComTrain_raw.tsv', sep='\t')
    data_test = pd.read_csv('drugsComTest_raw.tsv', sep='\t')

    return data_train, data_test

def df_train_test(data_train, data_test):
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


def reduce_conditions(value, train_df, test_df):
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


def cv(max_features=5000, stop_words=stopwords.words('english')):
    try:
        cv_train = pickle.load(open('cv_train.pkl', 'rb'))
        print('Data already exists.')
        print('Vocabulary length: ',cv_train.shape[1])

    except:
        data_train, data_test = read_data()
        train_df, test_df = df_train_test(data_train, data_test)
        reduce_conditions(50, train_df, test_df)

        y_train = train_df.condcopy
        y_test = test_df.condcopy

        ps = PorterStemmer()
        def tknizer(text):
            words = re.sub(r"[^A-Za-z0-9\-]", " ", text).lower().split()
            words = [ps.stem(word) for word in words if word not in stop_words]
            return words

        print('Initiating vectorizer...')
        cv = CountVectorizer(max_features=max_features, tokenizer=tknizer)
        print('vectorizer created\n')

        cv_train = cv.fit_transform(train_df.review)
        pickle.dump(cv_train, open('cv_train.pkl', 'wb'))
        print('cv_train saved')
        pickle.dump(y_train, open('y_train.pkl', 'wb'))
        print('y_train saved\n')
        cv_test = cv.transform(test_df.review)
        pickle.dump(cv_test, open('cv_test.pkl', 'wb'))
        print('cv_test saved')
        pickle.dump(y_test, open('y_test.pkl', 'wb'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Data preparation')
    parser.add_argument("--max_features", type=int)
    args=parser.parse_args()

    cv(args.max_features)