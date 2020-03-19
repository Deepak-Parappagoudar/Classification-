#!/usr/bin/env python
# coding: utf-8

# In[2]:



import pandas as pd
df = pd.read_csv('C:\\Users\\HP\\Desktop\\folders\\completed projects\\ML research classification\\datset.csv')
df.head()


# In[24]:


df = df[pd.notnull(df['Question'])]


# In[25]:


df.info()


# In[26]:


col = ['Question','Label','Label2']
df = df[col]


# In[27]:


df.columns


# In[28]:


df.columns=['Question','Label','Label2']


# In[29]:


df['category_id'] = df['Label'].factorize()[0]
df['category_id2'] = df['Label2'].factorize()[0]
from io import StringIO
category_id_df = df[['Label', 'category_id']].drop_duplicates().sort_values('category_id')
category_to_id = dict(category_id_df.values)
id_to_category = dict(category_id_df[['category_id', 'Label']].values)


category_id_df2 = df[['Label2', 'category_id2']].drop_duplicates().sort_values('category_id2')
category_to_id2 = dict(category_id_df2.values)
id_to_category2 = dict(category_id_df2[['category_id2', 'Label2']].values)


# In[ ]:





# In[30]:


df.head()


# In[31]:


# import matplotlib.pyplot as plt
# fig = plt.figure(figsize=(8,6))
# df.groupby('Label').Question.count().plot.bar(ylim=0)
# plt.show()


# In[32]:


from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')

features = tfidf.fit_transform(df.Question).toarray()
labels = df.category_id
features.shape


# In[33]:


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

X_train, X_test, y_train, y_test = train_test_split( df['Question'],df['Label'],test_size=0.25, random_state = 0)
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

clf = MultinomialNB().fit(X_train_tfidf, y_train)


X_train2, X_test2, y_train2, y_test2 = train_test_split( df['Question'],df['Label2'],test_size=0.25, random_state = 0)
count_vect2 = CountVectorizer()
X_train_counts2 = count_vect2.fit_transform(X_train2)
tfidf_transformer2 = TfidfTransformer()
X_train_tfidf2 = tfidf_transformer2.fit_transform(X_train_counts2)

clf2 = MultinomialNB().fit(X_train_tfidf2, y_train2)


# In[34]:


print(clf.predict(count_vect.transform(["tastiest fruit"])))
print(clf2.predict(count_vect2.transform(["tastiest fruit"])))


# In[35]:


df[df['Question'] == "tastiest fruit"]


# In[36]:



from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from sklearn.model_selection import cross_val_score


models = [
    RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0),
    LinearSVC(),
    MultinomialNB(),
    LogisticRegression(random_state=0),
]
CV = 5
cv_df = pd.DataFrame(index=range(CV * len(models)))
entries = []
for model in models:
  model_name = model.__class__.__name__
  accuracies = cross_val_score(model, features, labels, scoring='accuracy', cv=CV)
  for fold_idx, accuracy in enumerate(accuracies):
    entries.append((model_name, fold_idx, accuracy))
cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])


# In[37]:


# import seaborn as sns

# sns.boxplot(x='model_name', y='accuracy', data=cv_df)
# sns.stripplot(x='model_name', y='accuracy', data=cv_df, 
#               size=8, jitter=True, edgecolor="gray", linewidth=2)
# plt.show()


# In[38]:


cv_df.groupby('model_name').accuracy.mean()


# In[39]:


from sklearn.model_selection import train_test_split

model = LogisticRegression()

X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features, labels, df.index, test_size=0.25, random_state=0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

X_train2, X_test2, y_train2, y_test2, indices_train2, indices_test2 = train_test_split(features, labels, df.index, test_size=0.25, random_state=0)
model.fit(X_train2, y_train2)
y_pred2 = model.predict(X_test2)


# In[40]:


# from sklearn.metrics import confusion_matrix

# conf_mat = confusion_matrix(y_test, y_pred)
# fig, ax = plt.subplots(figsize=(8,8))
# sns.heatmap(conf_mat, annot=True, fmt='d',
#             xticklabels=category_id_df.Label.values, yticklabels=category_id_df.Label.values)
# plt.ylabel('Actual')
# plt.xlabel('Predicted')
# plt.show()


# In[41]:


from IPython.display import display

for predicted in category_id_df.category_id:
  for actual in category_id_df.category_id:
    if predicted != actual and conf_mat[actual, predicted] >= 6:
      print("'{}' predicted as '{}' : {} examples.".format(id_to_category[actual], id_to_category[predicted], conf_mat[actual, predicted]))
      display(df.loc[indices_test[(y_test == actual) & (y_pred == predicted)]][['Label', 'Question']])
      print('')
    
    


# In[42]:


texts = ["requiremnt of certification in agriculture","water content in orange fruit","types of agriculture"]
text_features = tfidf.transform(texts)
predictions = model.predict(text_features)
text_features2 = tfidf.transform(texts)
predictions2 = model.predict(text_features2)
for text, predicted ,predicted2 in zip(texts, predictions,predictions2):
  print('"{}"'.format(text))
  print("  - Predicted as: '{}'".format(id_to_category[predicted]))
  
  print("  - Predicted as: '{}'".format(id_to_category2[predicted2]))
  print("")

for text2, predicted2 in zip(texts, predictions2):
  
 


  print("")


# In[43]:


from sklearn import metrics
print(metrics.classification_report(y_test, y_pred, 
                                    target_names=df['Label'].unique()))

