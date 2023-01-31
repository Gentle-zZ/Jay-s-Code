import pandas as pd
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score

path_ham = "C:/Users/Jay/Documents/Code/Project/MATH620165/data/EmailClassifier/ham"
path_spam = "C:/Users/Jay/Documents/Code/Project/MATH620165/data/EmailClassifier/spam"
def readfile(path):
    files = os.listdir(path)
    s = []
    for file in files:
        f = open(path+"/"+file,encoding='gb18030',errors='ignore')
        iter_f = iter(f)
        str = ""
        for line in iter_f:
            str = str + line
        s.append(str)
    return s

s_ham = readfile(path_ham)

s_spam = readfile(path_spam)

label_ham = []
for i in range(len(s_ham)):
    label_ham.append("ham")

df_ham = pd.DataFrame({"cont":s_ham,
                         "label":label_ham})

label_spam = []
for i in range(len(s_spam)):
    label_spam.append("spam")

df_spam = pd.DataFrame({"cont":s_spam,
                         "label":label_spam})

data_init = pd.concat([df_ham,df_spam])

data_init['label'] = data_init.label.map({"ham":0,"spam":1})

X_train, X_test, y_train, y_test = train_test_split(data_init['cont'],data_init['label'],random_state=1)

count_vector = CountVectorizer(stop_words='english')
train_data = count_vector.fit_transform(X_train)
test_data = count_vector.transform(X_test)

naive_bayes = MultinomialNB()
naive_bayes.fit(train_data,y_train)
MultinomialNB(alpha=1.0,class_prior=None,fit_prior=True)
pred_nb = naive_bayes.predict(test_data)

print('Accuracy score:', format(accuracy_score(y_test, pred_nb)))
print('Precision score:', format(precision_score(y_test, pred_nb)))
print('Recall score:', format(recall_score(y_test, pred_nb)))
print('F1 score:', format(f1_score(y_test, pred_nb)))