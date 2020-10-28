
import numpy as np
import pandas as pd
import logging
import gensim 
from gensim import corpora
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
import numpy as np
from sklearn import metrics
from sklearn import svm
from sklearn.svm import SVC

# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# load train data:
def loadData(file1, file2, file3):
  twitter_economy = pd.read_csv(file1, encoding = 'utf-8')
  twitter_quarantine = pd.read_csv(file2, encoding = 'utf-8')
  twitter_vaccine = pd.read_csv(file3, encoding = 'utf-8')

  economy_data_text = twitter_economy['text']
  economy_data_label = twitter_economy['label']
  quarantine_data_text = twitter_quarantine['text']
  quarantine_data_label = twitter_quarantine['label']
  vaccine_data_text = twitter_vaccine['text']
  vaccine_data_label = twitter_vaccine['label']

  f_text = pd.concat( [economy_data_text, quarantine_data_text, vaccine_data_text], axis=0 )
  f_label = pd.concat( [economy_data_label, quarantine_data_label, vaccine_data_label], axis=0 )
  twitter = pd.concat( [f_text, f_label], axis=1)

  return twitter

def pre_process(text):
  mystopwords=stopwords.words("english") + ['one', 'the']
  WNlemma = nltk.WordNetLemmatizer()
  tokens = nltk.word_tokenize(text)
  tokens=[ WNlemma.lemmatize(t.lower()) for t in tokens]
  tokens=[ t for t in tokens if t not in mystopwords]
  tokens = [ t for t in tokens if len(t) >= 3 ]
  return(tokens)

def dtmGenerate(twitter):
  text = twitter['text']
  correct_labels = twitter['label']
  toks = text.apply(pre_process)
  dictionary = corpora.Dictionary(toks)
  dtm = [dictionary.doc2bow(d) for d in toks]
  return dtm

def Model(twitter):
  X_train, X_test, y_train, y_test = train_test_split(twitter.text, twitter.label, test_size=0.33, random_state=12)
  count_vect = CountVectorizer()
  text_clf = Pipeline([('vect', CountVectorizer()),   #Vectorizer
                     ('tfidf', TfidfTransformer()), #DTM with TFIDF
                      ('clf', svm.LinearSVC(C=1.0)),     #ML Model MultinomialNB() # svm.LinearSVC(C=1.0)
                    ])
  text_clf.fit(X_train.values.astype('U'), y_train.values.astype('U'))
  return text_clf, X_test, y_test

def predictedModel(text_clf, X_test, y_test):
  predicted = text_clf.predict(X_test.values.astype('U'))
  print(metrics.confusion_matrix(y_test.values.astype('U'), predicted))
  # print("NB:",np.mean(predicted == y_test.values.astype('U')))

twitter = loadData('topicModel/trainData/economy_new.csv', 'topicModel/trainData/quarantine_new.csv', 'topicModel/trainData/vaccine_new.csv')
text_clf, X_test, y_test = Model(twitter)
predictedModel(text_clf, X_test, y_test)
#
docs_new = ["Polio vaccines, given multiple times, can protect a child for life. Proper hygiene and sanitation"]
predicted = text_clf.predict(docs_new)
print(predicted)