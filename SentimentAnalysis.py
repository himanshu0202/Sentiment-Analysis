# coding: utf-8

# In[1]:

import re

import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsOneClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB

# In[2]:

df_romney = pd.read_excel('training-Obama-Romney-tweets.xlsx', sheetname='Romney')
df_obama = pd.read_excel('training-Obama-Romney-tweets.xlsx', sheetname='Obama')
inputFrame = df_romney
#print('Total number of tweets:' + str(len(inputFrame)))
#print('top few tweets:')
#inputFrame.head()


# In[5]:

def getInputFrame(df):
    inputFrame = df
    #inputFrame.dropna(inplace=True)
    inputFrame['Class'] = inputFrame['Class'].astype(str)
    inputFrame1 = inputFrame[inputFrame.Class == '1']
    inputFrame2 = inputFrame[inputFrame.Class == '-1']
    inputFrame3 = inputFrame[inputFrame.Class == '0']
    inputFrame_f = pd.concat([inputFrame1, inputFrame2, inputFrame3])
    return inputFrame_f


# In[6]:

tweetProcessFrame = getInputFrame(df_romney).copy()

# In[7]:

tweetProcessFrame.head()

# In[8]:

positiveWords, negWords = None, None

with open("positive-words.txt") as f:
    positiveWords = [ele.strip() for ele in f.readlines()]
with open("negative-words.txt",encoding = 'ISO-8859-1') as f:
    negWords = [ele.strip() for ele in f.readlines()]

# from nltk import PorterStemmer
# stemmer=PorterStemmer()
stop_words = set(stopwords.words('english'))
stemmer = SnowballStemmer("english")

emoticons = {}


# source : https://github.com/sifei/Dictionary-for-Sentiment-Analysis/tree/master/Emoticon
def initEmoticonDict():
    fileHandler = open('EmoticonsWithPolarity.txt', 'r')
    for line in fileHandler:
        emoticon_list = line[:-1].split(' ')
        sentiment = emoticon_list[-1]
        emoticon_list = emoticon_list[:-1]
        for emoticon in emoticon_list:
            emoticons[emoticon] = sentiment


initEmoticonDict()


def replaceWord(tweet, words, replacement):
    li = []
    for word in tweet:
        if word in words:
            li.append(replacement)
        else:
            li.append(word)
    return li


def stemWords(tweet):
    return [stemmer.stem(t) for t in tweet]


def mapEmoticons(tweet):
    list_words = tweet.split(' ')
    newtweet = ""
    for word in list_words:
        if word in emoticons:
            newtweet = newtweet + ' ' + emoticons.get(word)
        else:
            newtweet = newtweet + ' ' + word
    return newtweet


def removeTags(tweet):
    cleanr = re.compile('(</?[a-zA-Z]+>|https?:\/\/[^\s]*|(^|\s)RT(\s|$)|@[^\s]+|\d+)')
    cleantext = re.sub(cleanr, ' weblink ', tweet)
    cleantext = re.sub('(?<=^|(?<=[^a-zA-Z0-9-_\.]))@([A-Za-z]+[A-Za-z0-9]+)', ' usermention ', cleantext)
    cleantext = re.sub('[^\sa-zA-Z]+', '', cleantext)
    cleantext = re.sub('\s+', ' ', cleantext)
    return cleantext


def removePunctuations(tweet):
    exclude = set([',', '.', '!'])
    t = ''
    for ch in tweet:
        if ch not in exclude:
            t += ch
        else:
            t += ' '
    return tweet


def removeStopWords(tweet, stop_words):
    words_tokenize = word_tokenize(tweet)
    filtered_sentence = [w for w in words_tokenize if w not in stop_words]
    return filtered_sentence


# In[57]:

def clean(tweet):
    emotionFreeTweet = mapEmoticons(tweet)
    tagFreeTweet = removeTags(emotionFreeTweet)
    lowerCaseTweet = tagFreeTweet.lower()
    ptweet = replaceWord(lowerCaseTweet.split(' '), positiveWords, 'positive')
    ntweet = replaceWord(ptweet, negWords, 'negative')
    t = ' '.join(ntweet)
    puncFreeTweet = removePunctuations(t)
    stopFreeTweet = removeStopWords(puncFreeTweet, stop_words)
    finalTweet = stemWords(stopFreeTweet)
    return finalTweet


# In[ ]:

tweetProcessFrame.rename(columns={'date':'date','time':'time','Anootated tweet' : 'tweet','Class':'Class'},inplace = True)
tweetProcessFrame['tweet'] = tweetProcessFrame['tweet'].apply(clean)
del tweetProcessFrame['time']
del tweetProcessFrame['date']
def joinList1(tweetList):
    return " ".join(tweetList)

tweetProcessFrame['tweet'] = tweetProcessFrame['tweet'].apply(joinList1)
tweetProcessFrame1 = pd.DataFrame.drop_duplicates(tweetProcessFrame)
print('tweets after preprocessing')
tweetProcessFrame.head()


# spliting training and testing into 80-20 for testing

def splitTrainData(df, train_data_prcnt=80):
    msk = np.random.rand(len(df)) < train_data_prcnt/100
    train = df[msk]
    test = df[~msk]
    return train, test
tweet_random_df = tweetProcessFrame1.copy()

for i in range(0, 50):
    split1_df, split2_df = splitTrainData(tweet_random_df)
    tweet_random_df = pd.concat([split1_df, split2_df])
train, test = splitTrainData(tweet_random_df)


# In[ ]:

train_data = train['tweet']
train_label = train['Class']
train_label = pd.to_numeric(train_label)

test_data = test['tweet']
test_class = test['Class']
test_class = pd.to_numeric(test_class)


# In[ ]:

count_vect = CountVectorizer(max_features = 4800, ngram_range=(1, 2))
X_train_counts = count_vect.fit_transform(train_data)

tf_transformer = TfidfTransformer(use_idf=True).fit(X_train_counts)
X_train_tf = tf_transformer.transform(X_train_counts)


# Training Multinomial Naive Bayes


clf = MultinomialNB(alpha= 0.78, fit_prior= True).fit(X_train_tf, train_label)
X_test_counts = count_vect.transform(test_data)
X_test_tfidf = tf_transformer.transform(X_test_counts)
predicted = clf.predict(X_test_tfidf)

print('Test Accuracy:'+str(np.mean(predicted == test_class)))

print(metrics.classification_report(test_class, predicted))

print('Confusion Matrix:')
nb_confusion_matrix = metrics.confusion_matrix(test_class, predicted)
print(nb_confusion_matrix)

predicted_train = clf.predict(X_train_tf)
print('Train Accuracy:'+str(np.mean(predicted_train == train_label)))


# Support Vector Machine (Linear SVC)

text_clf_svm = Pipeline([('vect', CountVectorizer(max_features=4800,ngram_range=(1,2))),
                    ('tfidf', TfidfTransformer(use_idf=True)),
                    ('clf', LinearSVC(C=0.5,loss='hinge',multi_class='ovr',penalty='l2',random_state= 49, max_iter=10000))]).fit(train_data, train_label)

predicted_svm = text_clf_svm.predict(test_data)
print('Test Accuracy:'+str(np.mean(predicted_svm == test_class)))
print(metrics.classification_report(test_class, predicted_svm))

print('Confusion Matrix:')
linearSVM_confusion_matrix = metrics.confusion_matrix(test_class, predicted_svm)
print(linearSVM_confusion_matrix)

predicted_svm_train = text_clf_svm.predict(train_data)
print('Train Accuracy:'+str(np.mean(predicted_svm_train == train_label)))


# Voting classifier

# In[ ]:

clf_nb = MultinomialNB(alpha= 0.059, fit_prior= True)
clf_sgd = SGDClassifier(alpha=0.001,learning_rate='optimal',loss= 'epsilon_insensitive', penalty= 'l2',n_iter = 100, random_state=55)
clf_svc = LinearSVC(C = 0.6, loss = 'hinge', random_state= 55)
clf_rf = RandomForestClassifier(n_estimators = 22, class_weight = 'balanced_subsample', random_state = 55,criterion="gini")

voting_clf = Pipeline([('vect', CountVectorizer(max_features=4800,ngram_range=(1,2))),
                    ('tfidf', TfidfTransformer(use_idf= True)),
                    ('clf', VotingClassifier(estimators=[('mnb', clf_nb), ('sgd', clf_sgd), ('svm', clf_svc), ('rf',clf_rf)], voting='hard'))])

voting_clf = voting_clf.fit(train_data,train_label)

print("Voting Classifier")
p = voting_clf.predict(test_data)
print('Test Accuracy:'+str(np.mean(p==test_class)))
print(metrics.classification_report(test_class, p))

print('Confusion Matrix:')
voting_confusion_matrix = metrics.confusion_matrix(test_class,p)
print(voting_confusion_matrix)

predicted_eclf_train = voting_clf.predict(train_data)
print('Train Accuracy:'+str(np.mean(predicted_eclf_train == train_label)))

# SGD Classifier

print("Stochastic Gradient Descent")
text_clf_sgd = Pipeline([('vect', CountVectorizer(max_features=4800,ngram_range=(1,2))),
                    ('tfidf', TfidfTransformer(use_idf= True)),
                    ('clf', SGDClassifier(alpha=0.001,learning_rate='optimal',loss= 'epsilon_insensitive'
                                          ,penalty= 'l2',n_iter = 100, random_state=55))]).fit(train_data, train_label)

predicted_sgd = text_clf_sgd.predict(test_data)
print('Test Accuracy:'+str(np.mean(predicted_sgd == test_class)))
print(metrics.classification_report(test_class, predicted_sgd))

print('Confusion Matrix:')
linearSGD_confusion_matrix = metrics.confusion_matrix(test_class, predicted_sgd)
print(linearSGD_confusion_matrix)

predicted_sgd_train = text_clf_sgd.predict(train_data)
print('Train Accuracy:'+str(np.mean(predicted_sgd_train == train_label)))


# Random Forest


text_clf_RandomForest = Pipeline([('vect', CountVectorizer(max_features = 4800,ngram_range=(1,2))),
                    ('tfidf', TfidfTransformer(use_idf=True)),
                    ('clf', RandomForestClassifier(random_state = 36,criterion="gini",class_weight='balanced_subsample'
                                                   ,n_estimators=17))])
text_clf_RandomForest = text_clf_RandomForest.fit(train_data, train_label)

print("Random Forest")
predicted_rf = text_clf_RandomForest.predict(test_data)
print('Test Accuracy:'+str(np.mean(predicted_rf == test_class)))
print(metrics.classification_report(test_class, predicted_rf))

print('Confusion Matrix:')
linearRF_confusion_matrix = metrics.confusion_matrix(test_class, predicted_rf)
print(linearRF_confusion_matrix)

predicted_rf_train = text_clf_RandomForest.predict(train_data)

print('Train Accuracy:'+str(np.mean(predicted_rf_train == train_label)))

# OneVsOne Multiclass Learning
print("")
print("OneVsOne")
count_vect = CountVectorizer(max_features = 4800)
X_train_counts = count_vect.fit_transform(train_data)
tf_transformer = TfidfTransformer(use_idf=True).fit(X_train_counts)
X_train_tf = tf_transformer.transform(X_train_counts)

text_clf_OneVsOne = OneVsOneClassifier(LinearSVC(C = 0.5, loss = 'hinge', random_state= 55)).fit(X_train_tf, train_label)
X_train_counts = count_vect.transform(train_data)
X_train_tfidf = tf_transformer.transform(X_train_counts)
predicted = text_clf_OneVsOne.predict(X_train_tfidf)



print('Train Accuracy:'+str(np.mean(predicted == train_label)))

X_test_counts = count_vect.transform(test_data)
X_test_tfidf = tf_transformer.transform(X_test_counts)
predicted = text_clf_OneVsOne.predict(X_test_tfidf)

print('Test Accuracy:'+str(np.mean(predicted == test_class)))
print(metrics.classification_report(test_class, predicted))

print('Confusion Matrix:')
OneVsOne_confusion_matrix = metrics.confusion_matrix(test_class, predicted)
print(OneVsOne_confusion_matrix)



# Data Preparation for Cross Validation

# In[ ]:

combineFrameData = [train_data, test_data]
combineFrameLabel = [train_label, test_class]
combineTrainDataDF = pd.concat(combineFrameData)
combineTrainLabelDF = pd.concat(combineFrameLabel)
count_vect_kfold = CountVectorizer(max_features = 4800)
X_train_counts_kfold = count_vect.fit_transform(combineTrainDataDF)
tf_transformer_kfold = TfidfTransformer(use_idf=True).fit(X_train_counts_kfold)
X_train_tf_kfold = tf_transformer_kfold.transform(X_train_counts_kfold)


# K-fold accuracy for Naive Bayes

# In[ ]:
print("Naive Bayes")
clf_nb_kfold = MultinomialNB(alpha = 0.099, fit_prior = True)
scores = cross_val_score(clf_nb_kfold, X_train_tf_kfold, combineTrainLabelDF, cv=10)
clf_nb_kfold = clf_nb_kfold.fit(X_train_tf_kfold, combineTrainLabelDF)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
print(scores)


# K-fold accuracy for  Linear SVM

# In[ ]:
print("Linear SVM")
clf_lsvm_kfold = LinearSVC(C = 0.5, loss = 'hinge', penalty='l2', random_state= 42, max_iter=10000)
scores_lsvm = cross_val_score(clf_lsvm_kfold, X_train_tf_kfold, combineTrainLabelDF, cv=10)
clf_lsvm_kfold = clf_lsvm_kfold.fit(X_train_tf_kfold, combineTrainLabelDF)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores_lsvm.mean(), scores_lsvm.std() * 2))
print(scores_lsvm)


# K-fold accuracy for SGD

# In[ ]:
print("SGD")
clf_sgd_kfold = SGDClassifier(alpha=0.001,learning_rate='optimal',loss= 'epsilon_insensitive', penalty= 'l2',n_iter = 100, random_state=42)
scores_sgd = cross_val_score(clf_sgd_kfold, X_train_tf_kfold, combineTrainLabelDF, cv=10)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores_sgd.mean(), scores_sgd.std() * 2))
print(scores_sgd)


# K-fold accuracy for  Random Forest

# In[ ]:
print("Random Forest")
clf_randomForest_kfold = RandomForestClassifier(n_estimators = 22, class_weight = 'balanced_subsample', random_state = 42, criterion="gini")
scores_randomForest = cross_val_score(clf_randomForest_kfold, X_train_tf_kfold, combineTrainLabelDF, cv=10)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores_randomForest.mean(), scores_randomForest.std() * 2))
print(scores_randomForest)



# K-fold accuracy for OneVsOne
print("OneVsOne")
text_clf_OneVsOne = OneVsOneClassifier(LinearSVC(C = 0.5, loss = 'hinge', random_state= 55)).fit(X_train_tf, train_label)
scores_onevsone = cross_val_score(text_clf_OneVsOne, X_train_tf_kfold, combineTrainLabelDF, cv=10)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores_onevsone.mean(), scores_onevsone.std() * 2))
print(scores_onevsone)
