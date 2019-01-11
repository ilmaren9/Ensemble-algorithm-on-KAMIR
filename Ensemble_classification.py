# coding: utf-8

# This code is used in early prediction model on Korea Acute Myocardial infraction Patients (KAMIR) dataset

# This code enspired of: 
# Dataquest Data Science Blog - Introduction to Python Ensembles 
#         &
# Tutorial: Increasing the Predictive Power of Your Machine Learning Models with Stacking Ensembles

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


# In[27]:

kdata = pd.read_csv('data/KAMIR.csv', index_col='Id')
#y = pd.factorize(y.death_12)[0]
print(kdata.shape)
#train.death_12 = pd.factorize(train.death_12)[0]
#print(train.death_12)
#X, y = train.drop('death_12', axis=1), train.death_12.copy()
#train.head().iloc[:, :5]


# In[30]:

df = kdata.copy()


# In[32]: Label change object to vector

df.death_12 = pd.factorize(df.death_12)[0]


# In[42]:

print(df)


# In[37]:

df1 = df.select_dtypes(include=[np.number]).fillna(-1)
print(df1)


# In[39]: Label Encoder

from sklearn.preprocessing import LabelEncoder

df2 = df.copy()
for col in df2.columns:
    if df2[col].dtype == object:
        enc = LabelEncoder()
        df2[col] = enc.fit_transform(df[col].fillna('Missing'))

print('Dims', df2.shape)
df2.fillna(-1, inplace=True)


# In[40]:

df3 = df.copy()
cats = []
for col in df3.columns:
    if df3[col].dtype == object:
        df3 = df3.join(pd.get_dummies(df3[col], prefix=col), how='left')
        df3.drop(col, axis=1, inplace=True)
    

print('Dims', df3.shape)
df3.fillna(-1, inplace=True)


# A host of Scikit-learn models
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.kernel_approximation import Nystroem
from sklearn.kernel_approximation import RBFSampler
from sklearn.pipeline import make_pipeline


def get_models():
    """Generate a library of base learners."""
    svc = SVC(C=100, probability=True)
    knn = KNeighborsClassifier(n_neighbors=3)
    nn = MLPClassifier((80, 10), early_stopping=False, random_state=SEED)

    models = {'svm': svc,
              'knn': knn,
              'mlp-nn': nn
              }

    return models


def train_predict(model_list):
    """Fit models in list on training set and return preds"""
    P = np.zeros((ytest.shape[0], len(model_list)))
    P = pd.DataFrame(P)

    print("Fitting models.")
    cols = list()
    for i, (name, m) in enumerate(models.items()):
        print("%s..." % name, end=" ", flush=False)
        m.fit(xtrain, ytrain)
        P.iloc[:, i] = m.predict_proba(xtest)[:, 1]
        cols.append(name)
        print("done")

    P.columns = cols
    print("Done.\n")
    return P


def score_models(P, y):
    """Score model in prediction DF"""
    print("Scoring models.")
    for m in P.columns:
        score = roc_auc_score(y, P.loc[:, m])
        print("%-26s: %.3f" % (m, score))
        #print(f1_score(y, P.loc[:, m].round(), average="macro"))
        #print(precision_score(y, P.loc[:, m].round(), average="macro"))
        #print(recall_score(y, P.loc[:, m].round(), average="macro"))    
    print("Done.\n")

# In[118]:

from sklearn.metrics import accuracy_score

models = get_models()
P = train_predict(models)
score_models(P, ytest)
print("Accuracy", accuracy_score(ytest, P.mean(axis=1).round()))
print("Ensemble ROC-AUC score: %.3f" % roc_auc_score(ytest, P.mean(axis=1)))
print(f1_score(ytest, P.mean(axis=1).round(), average="macro"))
print(precision_score(ytest, P.mean(axis=1).round(), average="macro"))
print(recall_score(ytest, P.mean(axis=1).round(), average="macro"))    

models = get_models()
P = train_predict(models)
score_models(P, ytest)
print("Ensemble ROC-AUC score: %.3f" % roc_auc_score(ytest, P.mean(axis=1)))


# In[71]:

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_recall_fscore_support as score

knn = KNeighborsClassifier(n_neighbors=3)

knn.fit(xtrain, ytrain)
p = knn.predict_proba(xtest)[:, 1]
print("Average of decision tree ROC-AUC score: %.3f" % roc_auc_score(ytest, p))

precision, recall, fscore, support = score(ytest, p.round())

print(f1_score(ytest, p.round(), average="macro"))
print(precision_score(ytest, p.round(), average="macro"))
print(recall_score(ytest, p.round(), average="macro"))    


# In[153]:

from sklearn.metrics import roc_curve

def plot_roc_curve(ytest, P_base_learners, P_ensemble, labels, ens_label):
    """Plot the roc curve for base learners and ensemble."""
    plt.figure(figsize=(10, 8))
    plt.plot([0, 1], [0, 1], 'k--')
    
    cm = [plt.cm.rainbow(i)
      for i in np.linspace(0, 1.0, P_base_learners.shape[1] + 1)]
    
    for i in range(P_base_learners.shape[1]):
        p = P_base_learners[:, i]
        fpr, tpr, _ = roc_curve(ytest, p)
        plt.plot(fpr, tpr, label=labels[i], c=cm[i + 1])
        
    fpr, tpr, _ = roc_curve(ytest, P_ensemble)
    plt.plot(fpr, tpr, label=ens_label, c=cm[0])
        
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(frameon=False)
    plt.legend(loc="lower right")
    plt.show()


plot_roc_curve(ytest, P.values, P.mean(axis=1), list(P.columns), "proposed ensemble model")


# ## Ensemble

# In[73]:

base_learners = get_models()


# In[74]:

meta_learner = GradientBoostingClassifier(
    n_estimators=1000,
    loss="exponential",
    max_features=4,
    max_depth=3,
    subsample=0.5,
    learning_rate=0.005, 
    random_state=SEED
)


# In[75]:

xtrain_base, xpred_base, ytrain_base, ypred_base = train_test_split(
    xtrain, ytrain, test_size=0.5, random_state=SEED)


# In[76]:

def train_base_learners(base_learners, inp, out, verbose=True):
    """Train all base learners in the library."""
    if verbose: print("Fitting models.")
    for i, (name, m) in enumerate(base_learners.items()):
        if verbose: print("%s..." % name, end=" ", flush=False)
        m.fit(inp, out)
        if verbose: print("done")


# In[77]:

train_base_learners(base_learners, xtrain_base, ytrain_base)


# In[78]:

def predict_base_learners(pred_base_learners, inp, verbose=True):
    """Generate a prediction matrix."""
    P = np.zeros((inp.shape[0], len(pred_base_learners)))

    if verbose: print("Generating base learner predictions.")
    for i, (name, m) in enumerate(pred_base_learners.items()):
        if verbose: print("%s..." % name, end=" ", flush=False)
        p = m.predict_proba(inp)
        # With two classes, need only predictions for one class
        P[:, i] = p[:, 1]
        if verbose: print("done")

    return P


# In[79]:

P_base = predict_base_learners(base_learners, xpred_base)


# In[80]:

meta_learner.fit(P_base, ypred_base)


# In[81]:

def ensemble_predict(base_learners, meta_learner, inp, verbose=True):
    """Generate predictions from the ensemble."""
    P_pred = predict_base_learners(base_learners, inp, verbose=verbose)
    return P_pred, meta_learner.predict_proba(P_pred)[:, 1]


# In[82]:

P_pred, p = ensemble_predict(base_learners, meta_learner, xtest)
print("\nEnsemble ROC-AUC score: %.3f" % roc_auc_score(ytest, p))