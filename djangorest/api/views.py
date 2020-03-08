# from django.shortcuts import render

# Create your views here.
from rest_framework import generics
from django.shortcuts import get_object_or_404
from .serializers import mlmodelSerializer
from .models import mlmodel as mlModel
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from rest_framework.views import APIView
from rest_framework.response import Response
import re
import json
import datetime
import pytz
import os
import string
from pytz import timezone
from dateutil import tz

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from textblob import TextBlob

import matplotlib.pyplot as plt 

from sklearn.cross_validation import train_test_split

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from matplotlib.colors import ListedColormap
from sklearn.metrics import confusion_matrix

def NB(Xdata, Xtrain, ytrain):
    classifier = GaussianNB()
    classifier.fit(Xtrain, ytrain)
    y_pred = classifier.predict(Xdata)
    return(y_pred)


def LR(Xdata, Xtrain, ytrain):
    classifier = LogisticRegression(random_state = 0)
    classifier.fit(Xtrain, ytrain)
    y_pred = classifier.predict(Xdata)
    return(y_pred)


def KNN(Xdata, Xtrain, ytrain):
    classifier= KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
    classifier.fit(Xtrain, ytrain)
    y_pred = classifier.predict(Xdata)
    return(y_pred)


def SVM(Xdata, Xtrain, ytrain):
    classifier = SVC(kernel = 'linear', random_state = 0)
    classifier.fit(Xtrain, ytrain)
    y_pred = classifier.predict(Xdata) 
    return(y_pred)


def SVMrbf(Xdata, Xtrain, ytrain):
    classifier = SVC(kernel = 'rbf', random_state = 0)
    classifier.fit(Xtrain, ytrain)
    y_pred = classifier.predict(Xdata)
    return(y_pred) 


def DTC(Xdata, Xtrain, ytrain):
    classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
    classifier.fit(Xtrain, ytrain)
    y_pred = classifier.predict(Xdata)
    return(y_pred)


def RFC(Xdata, Xtrain, ytrain):
    classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
    classifier.fit(Xtrain, ytrain)
    y_pred = classifier.predict(Xdata)
    return(y_pred)



class api(APIView):

    def post(self, request):
        # var 
        twprofileArrlen = 0
        twstatusesArrlen = 0  
        alltweets = ''
        twLinkCount = 0 
        twQuestionCentric = 0 
        twReplyCount = 0 
        twTime = 0
        twNumUserMention = 0
        twRetweet = 0
        twCount =0 
        negativeCount = 0
        positivePercentage = 0
        neutralPercentage = 0 
        result = 0 
        from_zone = tz.gettz('UTC')
        to_zone = tz.gettz('Asia/Singapore')

        selectmodel = get_object_or_404(mlModel, id=1)

        twArrlen = len(request.data['twitter_profile'])
        twstatusesArrlen = len(request.data['twitter_statuses'])

        # tw
        tweetCount = request.data['twitter_profile'][twArrlen-1]['profile']['statuses_count']
        followersCount = request.data['twitter_profile'][twArrlen-1]['profile']['followers_count']
        followeesCount = request.data['twitter_profile'][twArrlen-1]['profile']['friends_count']
        favouriteCount = request.data['twitter_profile'][twArrlen-1]['profile']['favourites_count']
        twitterStatues = request.data['twitter_statuses'][twstatusesArrlen -1 ]['tweets']



        for tweets in twitterStatues:
            if('RT' in tweets['text']):
                twRetweet += 1
            if (not tweets['retweeted']) and ('RT @' not in tweets['text']):
                if(re.match(r'^[a-z0-9`\'\",/;:\(\)\[\]\$\&\s]+[\?]$' , tweets['text'], re.M |re.I )):
                    twQuestionCentric +=1
            
                if(tweets['entities']['urls']):
                    twLinkCount +=1
            
                if(tweets['created_at']):
                    date_str = tweets['created_at']
                    datetime_malaysia = datetime.datetime.strptime(date_str,"%a %b %d %H:%M:%S %z %Y" )
                    utc = datetime_malaysia.replace(tzinfo=from_zone)
                    central = utc.astimezone(to_zone)
                    twCovertMalaysia = central.hour 
                    if(twCovertMalaysia>20 and twCovertMalaysia<24):
                        twTime +=1
                    if(twCovertMalaysia>=0 and twCovertMalaysia<7):
                        twTime +=1
                twCount += 1

            alltweets += tweets['text']
        tokens = word_tokenize(alltweets)
        tokens = [w.lower() for w in tokens]
        table = str.maketrans('', '', string.punctuation)
        stripped = [w.translate(table) for w in tokens]
        words = [word for word in tokens if word.isalpha()]
        stop_words = set(stopwords.words('english'))
        words = [w for w in words if not w in stop_words]
        for word in words:
            print(word)
            analysis = TextBlob(word)
            if analysis.sentiment.polarity < 0:
                negativeCount += 1
                print(analysis)
                print('ne')
            if analysis.sentiment.polarity > 0: 
                positivePercentage +=1 
                print(analysis)
                print('po')
            if analysis.sentiment.polarity == 0:
                neutralPercentage +=1
                print(analysis)
                print('n')

        positivePercentage = (positivePercentage/len(words) *100 )
        neutralPercentage = (neutralPercentage/len(words) *100 )
        negativePercentage = (negativeCount/len(words))*100


        twTime = ((twTime/twCount) * 100)

        data = {'1': [followersCount],
        '2': [followeesCount],
        '3': [tweetCount],
        '4': [twQuestionCentric],
        '5': [twLinkCount],
        '6': [twTime],
        '7': [negativePercentage],
        '8': [twRetweet],
        }

        df = pd.DataFrame(data, index=['1'])
        print(df.values)

        twcwd = os.path.dirname(os.path.realpath(__file__))
        print(twcwd)
        filestring = twcwd+'\\'+selectmodel.csvfile
        dataset = pd.read_csv(filestring)
        X = dataset.iloc[:,0:8].values
        y = dataset.iloc[:,8].values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_input = sc.transform(df.values)

        if(selectmodel.name == 'LR'):
            result = LR(X_input, X_train, y_train)

        if(selectmodel.name == 'KNN'):
            result = KNN(X_input, X_train, y_train)

        if(selectmodel.name == 'SVM'):
            result = SVM(X_input, X_train, y_train)

        if(selectmodel.name == 'SVMrbf'):
            result = SVMrbf(X_input, X_train, y_train)

        if(selectmodel.name == 'NB'):
            result = NB(X_input, X_train, y_train)

        if(selectmodel.name == 'DTC'):
            result = DTC(X_input, X_train, y_train)

        if(selectmodel.name == 'RFC'):
            result = RFC(X_input, X_train, y_train)

        # # y_pred = classifier.predict(X_input)
        # y_test_pred = classifier.predict(X_test)
        # print(y_test)
        # print(y_test_pred)     
        # accuraries = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv =5)
        # print(accuraries.mean())

        # cm = confusion_matrix(y_test, y_test_pred)
        # print(cm)

        # models = []
        # models.append(('LDA', LinearDiscriminantAnalysis()))
        # models.append(('KNN', KNeighborsClassifier()))
        # models.append(('CART', DecisionTreeClassifier()))
        # models.append(('NB', GaussianNB()))
        # models.append(('SVM', SVC(kernel = 'linear')))

        # results = []
        # names = []
        # scoring = 'accuracy'
        # for name, model in models:
        #     kfold = model_selection.KFold(n_splits=5, random_state=10)
        #     cv_results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
        #     results.append(cv_results)
        #     names.append(name)
        #     msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        #     print(msg)
        # # boxplot algorithm comparison
        # fig = plt.figure()
        # fig.suptitle('Algorithm Comparison')
        # ax = fig.add_subplot(111)
        # plt.boxplot(results)
        # ax.set_xticklabels(names)
        # plt.show()

   

        # X_set, y_set = X_train, y_train
        # print(y_train)
        # X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
        #          np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
        # plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
        #      alpha = 0.75, cmap = ListedColormap(('red', 'green')))
        # plt.xlim(X1.min(), X1.max())
        # plt.ylim(X2.min(), X2.max())
        # for i, j in enumerate(np.unique(y_set)):
        #     plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
        #         c = ListedColormap(('red', 'green'))(i), label = j)
        # plt.title('SVM (Training set)')
        # plt.xlabel('Number of tweets')
        # plt.ylabel('Percentage of tweets that post at night')
        # plt.legend()
        # plt.show()

        


        # elimination 
        # name = request.data['facebook']['name']
        # common 
        # gender=request.data['facebook_profile']['gender']

        
        


        # fb 
        # print(to_zone)
        # print(gender)
        # print(tweetCount)
        # for tweets in twitterStatues:

            # if(re.match(r'\w+?$', tweets['text'], re.M|re.I)):
            #     twQuestionCentric +=1
            
            # if(tweets['entities']['urls']):
            #     twLinkCount +=1
            
            # if(tweets['retweet_count']):
            #     twRetweet += twRetweet
            
            # if(tweets['entities']['user_mentions']):
            #     twNumUserMention += len(tweets['entities']['user_mentions'])
            
            # if(tweets['created_at']):
            #     date_str = tweets['created_at']
            #     print(date_str)
            #     datetime_malaysia = datetime.datetime.strptime(date_str,"%a %b %d %H:%M:%S %z %Y" )
            #     print(type(datetime_malaysia))
            #     utc = datetime_malaysia.replace(tzinfo=from_zone)
            #     central = utc.astimezone(to_zone)
            #     print(central)
                # if(twCovertMalaysia>20 and twCovertMalaysia<24):
                #     twTime +=1;
                # if(twCovertMalaysia>=0 and twCovertMalaysia<7):
                #     twTime +=1;
                

        # print(twNumUserMention)
        # print(twLinkCount)
        # print(twQuestionCentric)
        # print(twRetweet)

        totalresult = {
            'result': result,
            'positive': positivePercentage,
            'negative': negativePercentage,
            'neutral' : neutralPercentage
        }

        return Response(totalresult)

class fbapi(APIView):
    def post(self, request):
        friendsCount = 0
        gender = 0
        feeds_count = 0 
        time = 0
        night_post = 0
        negative_post = 0 
        emoji_count = 0
        comment_count = 0 
        likes_count = 0 
        allfeeds = ''
        negativePercentage = 0;
        positivePercentage = 0;
        neutralPercentage =0;

        selectmodel = get_object_or_404(mlModel, id=1)
        result = 0 

        fbArrlen = len(request.data['facebook_profile'])
        fbfeedsArrlen = len(request.data['facebook_feed'])
        fbfriendsArrlen = len(request.data['facebook_friends'])

        friendsCount = request.data['facebook_friends'][fbfriendsArrlen-1]['friend']['summary']['total_count'] 
        if(request.data['facebook_profile'][fbArrlen-1]['profile']['gender'] == 'male'):
            gender = 1
        if(request.data['facebook_profile'][fbArrlen-1]['profile']['gender'] == 'female'):
            gender = 0

        for feed in request.data['facebook_feed'][fbfeedsArrlen-1]['feeds']:
            time = int(feed['created_time'][11:13])
            if(time > 20 and time <24):
                night_post +=1
            if(time >=0 and time <7):
                night_post +=1
            if 'reactions' in feed: 
                emoji_count += len(feed['reactions']['data'])
            if 'comments' in feed:
                comment_count += len(feed['comments']['data'])
            if 'likes' in feed:
                likes_count += len(feed['likes']['data'])
            if 'story' in feed:
                allfeeds += feed['story']
            if 'name' in feed:
                allfeeds += feed['name']
            if 'description' in feed:
                allfeeds += feed['description']
            if 'message' in feed:
                allfeeds += feed['message']

        token = word_tokenize(allfeeds)
        tokens = [w.lower() for w in token]
        table = str.maketrans('', '', string.punctuation)
        stripped = [w.translate(table) for w in tokens]
        words = [word for word in tokens if word.isalpha()]
        stop_words = set(stopwords.words('english'))
        words = [w for w in words if not w in stop_words]
        for word in words:
            analysis = TextBlob(word)
            if analysis.sentiment.polarity < 0:
                negative_post += 1
                print(analysis)
                print('ne')
            if analysis.sentiment.polarity > 0: 
                positivePercentage +=1
                print(analysis)
                print('po')
            if analysis.sentiment.polarity == 0:
                neutralPercentage +=1
                print(analysis)
                print('n')

        if(positivePercentage != 0):
            positivePercentage = (positivePercentage/len(words) *100 )
        else:
            positivePercentage = 0
        
        if(neutralPercentage != 0):
            neutralPercentage = (neutralPercentage/len(words) *100 )
        else:
            neutralPercentage = 0
        
        if(negativePercentage != 0):
            negativePercentage = (negative_post/len(words)*100)
        else:
            negativePercentage = 0
        

        data = {'1': [gender],
        '2': [friendsCount],
        '3': [feeds_count],
        '4': [night_post],
        '5': [negative_post],
        '6': [emoji_count],
        '7': [comment_count],
        '8': [likes_count],
        }

        df = pd.DataFrame(data, index=['1'])
        print(df.values)

        fbcwd = os.path.dirname(os.path.realpath(__file__))
        print(fbcwd)
        filestring = fbcwd+'\\'+'fbtrainingdata.csv'
        dataset = pd.read_csv(filestring)
        X = dataset.iloc[:,0:8].values
        y = dataset.iloc[:,8].values

        print(X)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_input = sc.transform(df.values)

        if(selectmodel.name == 'LR'):
            result = LR(X_input, X_train, y_train)

        if(selectmodel.name == 'KNN'):
            result = KNN(X_input, X_train, y_train)

        if(selectmodel.name == 'SVM'):
            result = SVM(X_input, X_train, y_train)

        if(selectmodel.name == 'SVMrbf'):
            result = SVMrbf(X_input, X_train, y_train)

        if(selectmodel.name == 'NB'):
            result = NB(X_input, X_train, y_train)

        if(selectmodel.name == 'DTC'):
            result = DTC(X_input, X_train, y_train)

        if(selectmodel.name == 'RFC'):
            result = RFC(X_input, X_train, y_train)


        totalresult = {
            'result': result,
            'positive': positivePercentage,
            'negative': negativePercentage,
            'neutral' : neutralPercentage
        }

        return Response(totalresult)



class mlmodel(APIView):
    def get(self, request):
        mlmodel1 = get_object_or_404(mlModel, id=1)
        serializer = mlmodelSerializer(mlmodel1, many=True)
        return Response(serializer.data)
    
    def post(self,request):
        # data = {
        #  accuracy : 0,
        #  precision : 0,
        #  f1score : 0,
        #  recall : 0,
        #  truepositiverate : 0,
        #  falsepositiverate : 0, 

        # }

        selectmodel = get_object_or_404(mlModel, id=1)
        print(request.data)
        selectmodel.name = request.data['ml']
        selectmodel.csvfile = request.data['file']
        filestring = 'C:\\Users\\User\\Desktop\\DjangoRest\\djangorest\\api\\{}'.format(selectmodel.csvfile)
        dataset = pd.read_csv(filestring)
        X = dataset.iloc[:,0:8].values
        y = dataset.iloc[:,8].values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)

        if(selectmodel.name == 'LR'):
            classifier = LogisticRegression(random_state = 0)
            classifier.fit(X_train, y_train)
            y_pred = classifier.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)

        if(selectmodel.name == 'KNN'):
            classifier= KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
            classifier.fit(X_train, y_train)
            y_pred = classifier.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)

        if(selectmodel.name == 'SVM'):
            classifier = SVC(kernel = 'linear', random_state = 0)
            classifier.fit(X_train, y_train)
            y_pred = classifier.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)

        if(selectmodel.name == 'SVMrbf'):
            classifier = SVC(kernel = 'rbf', random_state = 0)
            classifier.fit(X_train, y_train)
            y_pred = classifier.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)

        if(selectmodel.name == 'NB'):
            classifier = GaussianNB()
            classifier.fit(X_train, y_train)
            y_pred = classifier.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)

        if(selectmodel.name == 'DTC'):
            classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
            classifier.fit(X_train, y_train)
            y_pred = classifier.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)

        if(selectmodel.name == 'RFC'):
            classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
            classifier.fit(X_train, y_train)
            y_pred = classifier.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)

        print(cm)

        selectmodel.tn = cm[0][0]
        selectmodel.fp = cm[0][1]
        selectmodel.fn = cm[1][0]
        selectmodel.tp = cm[1][1]
        total = selectmodel.tn + selectmodel.fp + selectmodel.fn  + selectmodel.tp

        selectmodel.accuracy = (selectmodel.tp + selectmodel.tn)/ total
        selectmodel.truepositiverate = (selectmodel.tp /(selectmodel.fn + selectmodel.tp))
        selectmodel.falsepositiverate = (selectmodel.fp/(selectmodel.tn + selectmodel.fp))
        selectmodel.precision = (selectmodel.tp/(selectmodel.fp + selectmodel.tp))
        selectmodel.recall = (selectmodel.tp /(selectmodel.tp+selectmodel.fn))
        selectmodel.f1score = ((2*selectmodel.recall*selectmodel.precision)/(selectmodel.recall + selectmodel.precision))
        selectmodel.save()

        data = {
         'accuracy' : round(selectmodel.accuracy,3),
         'precision' : round(selectmodel.precision,3),
         'f1score' : round(selectmodel.f1score,3),
         'recall' : round(selectmodel.recall,3),
         'truepositiverate' : round(selectmodel.truepositiverate,3),
         'falsepositiverate' : round(selectmodel.falsepositiverate,3),

        }

        return Response(data)