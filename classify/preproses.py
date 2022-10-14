import nltk
import string
import re
import requests
import numpy
import pymysql
import pandas as pd
import numpy as np
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from sklearn import model_selection

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mutual_info_score
from tweepy import OAuthHandler
import tweepy

consumerKey = "aOeKsPYiHaNogKz839cOjNcrw"
consumerSecret = "7iVB2Wrmx5HgaDpKvEpQuOvSHSEbQvLPDRYlFBLWwLU3iQN8uh"
accessToken = "456738617-OdAQgKaDCsMpv3V2Ky20lhiphIqjGDQjbrpEAJ6v"
accessTokenSecret = "LalwuqFdMEi1CgC6t4GfQvn0J50ittqllZ2Uha6W3mPX4"
auth = OAuthHandler(consumerKey, consumerSecret)
auth.set_access_token(accessToken, accessTokenSecret)
api = tweepy.API(auth)


def bacafile(tweet):
    # tweet = tweet.str.replace('b+', '')
    remove_url = tweet.str.replace('http\S+|https\S+|www.\S+', '')
    remove_rt = remove_url.str.replace('RT+', '')
    remove_mention = remove_rt.str.replace('@[A-Za-z0-9|A_Za_z0_9]+', '')
    remove_hashtag = remove_mention.str.replace('#[A-Za-z0-9|A_Za_z0_9]+', '')
    remove_number = remove_hashtag.str.replace('\d+', '')
    remove_punctuation = remove_number.str.replace(
        '[{}]'.format(string.punctuation), '')
    lower_case = remove_punctuation.str.lower()
    lower_case = lower_case.str.replace('\ufeff', '')
    lower_case = lower_case.str.replace('\n', ' ')
    lower_case = lower_case.str.replace('\r', ' ')
    lower_case = lower_case.str.replace('\n\n', ' ')
    lower_case = lower_case.str.replace('\r\r', ' ')
    return lower_case


def bacafile_uji(tweet):
    # tweet = tweet.str.replace('b+', '')
    remove_url = tweet.replace('http\S+|https\S+|www.\S+', '')
    remove_rt = remove_url.replace('RT+', '')
    remove_mention = remove_rt.replace('@[A-Za-z0-9|A_Za_z0_9]+', '')
    remove_hashtag = remove_mention.replace('#[A-Za-z0-9|A_Za_z0_9]+', '')
    remove_number = remove_hashtag.replace('\d+', '')
    remove_punctuation = remove_number.replace(
        '[{}]'.format(string.punctuation), '')
    lower_case = remove_punctuation.lower()
    lower_case = lower_case.replace(',', '')
    lower_case = lower_case.replace('-', '')
    lower_case = lower_case.replace('\ufeff', '')
    lower_case = lower_case.replace('\n', ' ')
    lower_case = lower_case.replace('\r', ' ')
    lower_case = lower_case.replace('\n\n', ' ')
    lower_case = lower_case.replace('\r\r', ' ')
    return lower_case


def stem(tweets):
    factorys = StopWordRemoverFactory()
    stopword = factorys.create_stop_word_remover()

    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    result = []
    for sentence in tweets:
        #         result.append(stemmer.stem(stopword.remove(sentence)))
        stop = stopword.remove(sentence)
        result.append(stemmer.stem(stop))
    return result


def stem_uji(tweets):
    factorys = StopWordRemoverFactory()
    stopword = factorys.create_stop_word_remover()

    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    result = []
    stop = stopword.remove(tweets)
    result.append(stemmer.stem(stop))
    return result


def random():
    x = []
    y = []
    connection = pymysql.connect(host='localhost',
                                 user='root',
                                 password='',
                                 db='klasifikasi_teks')

    # create cursor
    cursor = connection.cursor()

    # Execute query
    sql = "SELECT * FROM `preproses`"
    cursor.execute(sql)

    # Fetch all the records
    result = cursor.fetchall()
    for row in result:
        x.append(row[0])
        y.append(row[2])

    x = np.array(x)
    y = np.array(y)

    x_train, x_test, y_train, y_test = model_selection.train_test_split(
        x, y, test_size=0.25, random_state=0)

    return x_test


def computeSentimentStats(tweetSentimentPairs):
    totalMar = 0.0
    totalSed = 0.0
    totalSen = 0.0
    for pair in tweetSentimentPairs:
        if (pair == 3):
            totalMar += 1
        elif (pair == 2):
            totalSed += 1
        else:
            totalSen += 1
    total = totalSen+totalMar+totalSed

    return [round(100*(totalMar/total), 3), round(100*(totalSed/total), 3), round(100*(totalSen/total), 3)]


def getTweets(n, contains):
    """returns n tweets that contain the contains parameter, but with that string removed from the tweet for classification purposes"""
    tweets = []
    i = 0
    for tweet in tweepy.Cursor(api.search,
                               q=contains + "-filter:retweets",
                               rpp=100,
                               result_type="mixed",
                               include_entities=True,
                               lang="in").items():
        # Replace the searched term so it is not used in sentiment classification
        tweetText = tweet.text
        """Remove links and no traditional characters from a tweet"""
    # Remove links
        tweets.append(tweetText)
        i += 1
        if i >= n:
            break
    return tweets


def klasifikasi_train():
    connection = pymysql.connect(host='localhost',
                                 user='root',
                                 password='',
                                 db='klasifikasi_teks')

    # create cursor
    cursor = connection.cursor()

    # Execute query
    sql = "SELECT * FROM `training_data`"
    cursor.execute(sql)

    # Fetch all the records
    result = cursor.fetchall()

    return result


def random():
    x = []
    y = []
    connection = pymysql.connect(host='localhost',
                                 user='root',
                                 password='',
                                 db='klasifikasi_teks')

    # create cursor
    cursor = connection.cursor()

    # Execute query
    sql = "SELECT * FROM `validasi_data`"
    cursor.execute(sql)

    # Fetch all the records
    result = cursor.fetchall()
    for row in result:
        x.append(row[1])
        y.append(row[2])

    x = np.array(x)
    y = np.array(y)

    x_train, x_test, y_train, y_test = model_selection.train_test_split(
        x, y, test_size=0.25, random_state=0)

    return x_test


def klasifikasi_validasi():
    connection = pymysql.connect(host='localhost',
                                 user='root',
                                 password='',
                                 db='klasifikasi_teks')

    # create cursor
    cursor = connection.cursor()

    # Execute query
    sql = "SELECT * FROM `validasi_data`"
    cursor.execute(sql)

    # Fetch all the records
    result = cursor.fetchall()

    return result


def knn(x, y, x_test):
    text_clf = KNeighborsClassifier(n_neighbors=5)
    text_clf = text_clf.fit(x, y)
    y_pred_knn = text_clf.predict(x_test)
    return y_pred_knn

# ### Mutual Information digunakan untuk menghitung Mutual Information dari data setelah Count Vect


def mutual_information(x_train, y_train, ambil, x_test):
    # n - data
    # b - label

    x_train = np.array(x_train.todense())
    x_test = np.array(x_test.todense())

    x_train = np.array(x_train)
    x_test = np.array(x_test)

    a = x_train.shape[1]

    mi = np.zeros((a, 1))
    for i in range(a):
        mi[i] = mutual_info_score(x_train[:, i].ravel(), y_train)

    sorted_mix = sorted(range(len(mi)), key=lambda l: mi[l], reverse=True)
    sorted_mi = sorted_mix[:ambil]

    x_train_mi = np.zeros((len(x_train), ambil))
    x_test_mi = np.zeros((len(x_test), ambil))

    for i in range(len(sorted_mi)):
        x_train_mi[:, i] = x_train[:, sorted_mi[i]]

        x_test_mi[:, i] = x_test[:, sorted_mi[i]]
    return x_train_mi, x_test_mi


def confusion_matrix():
    connection = pymysql.connect(host='localhost',
                                 user='root',
                                 password='',
                                 db='klasifikasi_teks')

    # create cursor
    cursor = connection.cursor()

    # Execute query
    sql = "SELECT * FROM `confusion_matrix`"
    cursor.execute(sql)

    # Fetch all the records
    result = cursor.fetchall()

    return result


def confusion_matrix_validasi():
    connection = pymysql.connect(host='localhost',
                                 user='root',
                                 password='',
                                 db='klasifikasi_teks')

    # create cursor
    cursor = connection.cursor()

    # Execute query
    sql = "SELECT * FROM `confusion_matrix_validasi`"
    cursor.execute(sql)

    # Fetch all the records
    result = cursor.fetchall()

    return result
