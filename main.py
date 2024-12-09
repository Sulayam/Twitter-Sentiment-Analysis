import numpy as numpy
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

train_df = pd.read_csv('train1.csv')
test_df = pd.read_csv('test1.csv')

train_df.head()

train_df.info()

print("Shape of train_df:",train_df.shape)
print("Shape of test_df:", test_df.shape)

Columns to consider dropping - Flag, date, id

# prompt: print count of target=1 and target=0

target_counts = train_df['target'].value_counts()
print("Count of target=1:", target_counts[1])
print("Count of target=0:", target_counts[0])


train_df.drop(['flag', 'date', 'id', 'user'], axis=1, inplace=True)
test_df.drop(['flag', 'date', 'id', 'user'], axis=1, inplace=True)

train_df.head()

train_df.isnull().sum()

test_df.shape

import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Assuming 'text' is the column name in the train_df containing the reviews
corpus = []

# Iterate over all rows in train_df
for i in range(len(train_df)):
    review = re.sub('[^a-zA-Z]', ' ', train_df['text'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    all_stopwords = stopwords.words('english')
    if 'not' in all_stopwords:
        all_stopwords.remove('not')
    review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
    review = ' '.join(review)
    corpus.append(review)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 9000)
X = cv.fit_transform(corpus).toarray()
y = train_df.iloc[:, -1].values

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X, y)

test_ids = pd.read_csv('test1.csv')['id']

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer

# Assuming 'text' is the column name in test_df
corpus_test = []

# Iterate over all rows in test_df
for i in range(len(test_df)):
    review = re.sub('[^a-zA-Z]', ' ', test_df['text'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    all_stopwords = stopwords.words('english')
    if 'not' in all_stopwords:
        all_stopwords.remove('not')
    review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
    review = ' '.join(review)
    corpus_test.append(review)

# Use the same CountVectorizer (cv) fitted on the training data
X_test = cv.transform(corpus_test).toarray()  # Note: use transform, not fit_transform

# Now you can predict probabilities
predictions1 = classifier.predict_proba(X_test)

predictions = predictions1[:,1]

print(predictions)

predictions=[1 if x>=0.5 else 0 for x in predictions]
results=pd.DataFrame({'id': test_ids, 'Target':predictions})
results.to_csv('Team_1_baseline_bag_of_words_0.2.csv',index=False)

results.head()
