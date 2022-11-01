import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import base64
import io
import matplotlib.pyplot as pyplot
from matplotlib.figure import Figure
from django.shortcuts import render
from classify import preproses as preproses
import requests
import pymysql
from sqlalchemy import create_engine
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import model_selection
from sklearn import metrics
from .forms import SearchForm
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sn
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt


context = {
    'input': '',
    'input_split': '',
    'output_norm': '',
}

# from .forms import SearchForm

matplotlib.use('Agg')
plt.rcdefaults()

# Create your views here.


def home(request):
    return render(request, "home.html")


def training(request):
    result = []
    if request.method == 'POST':
        file = request.FILES['file']
        df = pd.read_json(file)
        tweet = df['x_train']
        lower = preproses.bacafile(tweet)
        stemm = preproses.stem(lower)
        sentimen_manual = df['y_train']

        dict = {'tweet': tweet,
                'stem': stemm, 'sentimen_manual': sentimen_manual}
        df = pd.DataFrame(dict)
        engine = create_engine(
            'mysql+pymysql://root:@localhost/klasifikasi_teks')
        df.to_sql('training_data', con=engine,
                  if_exists='replace', index=False)

        connection = pymysql.connect(host='localhost',
                                     user='root',
                                     password='',
                                     db='klasifikasi_teks')

        # create cursor
        cursor = connection.cursor()

        cursor = connection.cursor()
        sql = "SELECT * FROM `training_data`"
        cursor.execute(sql)

        result = cursor.fetchall()

    return render(request, "training.html", {'result': result})


def validasi(request):
    fix_result = []
    response = []
    result_cm = []
    mi_feature = 500
    x = []
    y = []
    xtrain = []
    ytrain = []

    y_pred = []

    if request.method == 'POST':
        file = request.FILES['file']
        df = pd.read_json(file)

        tweet = df['x_train']
        lower = preproses.bacafile(tweet)
        stemm = preproses.stem(lower)
        sentimen_manual = df['y_train']

        dict = {'tweet': tweet,
                'stem': stemm, 'sentimen_manual': sentimen_manual}
        df = pd.DataFrame(dict)
        engine = create_engine(
            'mysql+pymysql://root:@localhost/klasifikasi_teks')
        df.to_sql('validasi_data', con=engine,
                  if_exists='replace', index=False)

        connection = pymysql.connect(host='localhost',
                                     user='root',
                                     password='',
                                     db='klasifikasi_teks')

        cursor = connection.cursor()
        sql = "SELECT * FROM `validasi_data`"
        cursor.execute(sql)

        result = cursor.fetchall()
        for row in result:
            x.append(row[1])
            y.append(row[2])
        x_test = np.array(x)
        y_test = np.array(y)

        resultTrain = preproses.klasifikasi_train()
        for row in resultTrain:
            xtrain.append(row[1])
            ytrain.append(row[2])
        x_train = np.array(xtrain)
        y_train = np.array(ytrain)

        countvect = CountVectorizer()
        c = countvect.fit(x_train)
        x_mi_train = c.transform(x_train)
        x_mi_test = c.transform(x_test)

        x_mi_train, x_mi_test = preproses.mutual_information(
            x_mi_train, y_train, mi_feature, x_mi_test)

        clf = KNeighborsClassifier(n_neighbors=39)
        clf.fit(x_mi_train, y_train)
        predictions = clf.predict(x_mi_test)
        cr_y1 = classification_report(y_test, predictions)
        print(cr_y1)

        xdata = preproses.validasi_data()
        df = pd.DataFrame()
        df['xdata'] = xdata
        df['x_test'] = x_test
        df['y_test'] = y_test

        dict = {'tweet': xdata, 'stem': df['x_test'],
                'sentimen_manual': df['y_test']}
        df = pd.DataFrame(dict)
        engine = create_engine(
            'mysql+pymysql://root:@localhost/klasifikasi_teks')
        df.to_sql('validasi_data_result', con=engine,
                  if_exists='replace', index=False)

        # precision = df['precision']
        # print(df['-1'])

        # print(metrics.classification_report(y_test, y_pred))

        knn_cv = KNeighborsClassifier(n_neighbors=39)
        cv_scores = cross_val_score(knn_cv, X=x_mi_test, y=y_test, cv=10)
        # print(cv_scores)
        # print("cv_scores mean:{}".format(np.mean(cv_scores)))

        num = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
        dframe = pd.DataFrame(cv_scores, columns=['scores'])

        dframe['fold'] = num

        data_kcrossvalidation = dframe[['fold', 'scores']]

        df = pd.DataFrame(data_kcrossvalidation)
        engine = create_engine(
            'mysql+pymysql://root:@localhost/klasifikasi_teks')
        df.to_sql('data_kcrossvalidation', con=engine,
                  if_exists='replace', index=False)

        result_cm = preproses.data_kcrossvalidation()

        fold = dframe['fold']
        scores = dframe['scores']

        plt.title('Scores by K-Cross Validation')
        plt.xlabel('Num Cross Validation')
        plt.ylabel('Scores')
        plt.plot(fold, scores)
        plt.show()
        # fig = Figure()
        # plt.axis('image')
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        # plt.close(fig)
        response = base64.b64encode(buf.getvalue()).decode(
            'utf-8').replace('\n', '')
        buf.close()

        for i in range(len(y_test)):
            fix_result.append([tweet[i], x_test[i], y_test[i]])

    return render(request, "validasi.html", {'result': fix_result,  'result_cm': result_cm, 'response': response})
    # return render(request, "testing.html", {'result': fix_result, 'response': response})


def testing(request):
    x_test = []
    y_test = []
    x_train = []
    y_train = []
    y_test = []
    fix_result = []
    percent_sen = "0"
    percent_sed = "0"
    percent_mar = "0"
    response = []
    result_cm = []
    mi_feature = 500

    pred = []

    if request.method == 'POST':
        file = request.FILES['file']
        df = pd.read_json(file)

        tweet = df['x_test']
        lower = preproses.bacafile(tweet)
        stemm = preproses.stem(lower)
        sentimen_manual = df['y_test']

        dict = {'tweet': tweet,
                'stem': stemm, 'sentimen_manual': sentimen_manual}
        df = pd.DataFrame(dict)
        engine = create_engine(
            'mysql+pymysql://root:@localhost/klasifikasi_teks')
        df.to_sql('testing_data', con=engine,
                  if_exists='replace', index=False)

        connection = pymysql.connect(host='localhost',
                                     user='root',
                                     password='',
                                     db='klasifikasi_teks')

        cursor = connection.cursor()
        sql = "SELECT * FROM `testing_data`"
        cursor.execute(sql)

        result = cursor.fetchall()
        for row in result:
            x_test.append(row[1])
            y_test.append(row[2])
        x_test = np.array(x_test)
        y_test = np.array(y_test)

        xdata = preproses.klasifikasi_train()
        for row in xdata:
            x_train.append(row[1])
            y_train.append(row[2])
        x_train = np.array(x_train)
        y_train = np.array(y_train)

        countvect = CountVectorizer()
        c = countvect.fit(x_train)
        x_mi_train = c.transform(x_train)
        x_mi_test = c.transform(x_test)

        x_mi_train, x_mi_test = preproses.mutual_information(
            x_mi_train, y_train, mi_feature, x_mi_test)

        y_pred = preproses.knn(x_mi_train, y_train, x_mi_test)

        dict = {'tweet': tweet,
                'stem': stemm, 'sentimen_manual': sentimen_manual, 'sentimen_knn': y_pred}
        df = pd.DataFrame(dict)
        engine = create_engine(
            'mysql+pymysql://root:@localhost/klasifikasi_teks')
        df.to_sql('testing_data_result', con=engine,
                  if_exists='replace', index=False)

        acc = metrics.accuracy_score(y_test, y_pred)
        print('accuracy = '+str(acc*100)+'%')
        print(metrics.classification_report(y_test, y_pred))

        report_dict = metrics.classification_report(
            y_test, y_pred, output_dict=True)
        df = pd.DataFrame(report_dict)
        engine = create_engine(
            'mysql+pymysql://root:@localhost/klasifikasi_teks')
        df.to_sql('confusion_matrix', con=engine,
                  if_exists='replace', index=False)

        result_cm = preproses.confusion_matrix()

        # precision = df['precision']
        # print(df['-1'])

        # print(metrics.classification_report(y_test, y_pred))

        labels = ["senang", "sedih", "marah"]
        cf_matrix = confusion_matrix(y_test, y_pred)
        ax = sn.heatmap(cf_matrix, annot=True, xticklabels=labels,
                        yticklabels=labels, cmap="YlGnBu", fmt='g')
        plt.xlabel('Prediction')
        plt.ylabel('Actual')
        fig = Figure()
        plt.axis('image')
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)
        response = base64.b64encode(buf.getvalue()).decode(
            'utf-8').replace('\n', '')
        buf.close()

        for i in range(len(y_pred)):
            fix_result.append([tweet[i], x_test[i], y_test[i], y_pred[i]])

        for a in fix_result:
            pred.append(a[3])
            print(pred)
            sentimentStats = preproses.computeSentimentStats(pred)
            percent_mar = sentimentStats[0]
            percent_sed = sentimentStats[1]
            percent_sen = sentimentStats[2]

    return render(request, "testing.html", {'result': fix_result,  'result_cm': result_cm, 'percent_sen': percent_sen, 'percent_sed': percent_sed, 'percent_mar': percent_mar, 'response': response})
    # return render(request, "testing.html", {'result': fix_result, 'response': response})


def ujidata(request):
    x_train = []
    y_train = []
    mi_feature = 500
    y_pred = []
    text = []
    pred = []
    fix_result = []

    if request.method == 'GET':
        pass
    elif request.method == 'POST':
        text = request.POST['your-text']
        # classifier_ = clasy.testClassifier(text_)
        bacafile = preproses.bacafile_uji(text)
        stem = preproses.stem_uji(bacafile)
        x_test = np.array(stem)

        xdata = preproses.klasifikasi_train()
        for row in xdata:
            x_train.append(row[1])
            y_train.append(row[2])
        x_train = np.array(x_train)
        y_train = np.array(y_train)

        countvect = CountVectorizer()
        c = countvect.fit(x_train)
        x_mi_train = c.transform(x_train)
        x_mi_test = c.transform(x_test)

        x_mi_train, x_mi_test = preproses.mutual_information(
            x_mi_train, y_train, mi_feature, x_mi_test)

        y_pred = preproses.knn(x_mi_train, y_train, x_mi_test)

        context['input'] = text
        context['output_norm'] = y_pred

    return render(request, 'uji.html', context)
