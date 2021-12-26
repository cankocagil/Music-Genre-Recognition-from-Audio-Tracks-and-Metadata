# %%
from __future__ import (
    print_function,
    division
)

import matplotlib.pyplot as plt 
import seaborn as sns 
import pandas as pd
import numpy as np
import feather
import pickle
import random
import glob
import time
import sys
sys.path.append('../src')
import os


from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    precision_score,
    roc_auc_score
)

from typing import (
    Callable,
    Iterable,
    List
)

from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler
)

from tqdm.notebook import tqdm_notebook

from tqdm import tqdm

from sklearn.utils.validation import (
    check_X_y, 
    check_array
)

tqdm_notebook.pandas()

# %% [markdown]
# # Utility Functions

# %%
def confusion_matrix(
    labels:Iterable[list or np.ndarray],
    preds:Iterable[list or np.ndarray]
    ) -> pd.DataFrame:
    """ Takes desireds/labels and softmax predictions, return a confusion matrix. """
    label = pd.Series(
        labels,
        name='Actual'
    )
    pred = pd.Series(
        preds,
        name='Predicted'
    )
    return pd.crosstab(
        label,
        pred
    )


def visualize_confusion_matrix(
    data:np.ndarray,
    normalize:bool = True,
    title:str = " "
) -> None:
    
    if normalize:
        data /= np.sum(data)

    plt.figure(figsize=(15,15))
    sns.heatmap(data, 
                fmt='.2%',
                cmap = 'Greens')

    plt.title(title)
    plt.show()


def save_obj(obj:object, path:str = None) -> None:
    with open(path + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

        
def load_obj(path:str = None) -> object:
    with open(path + '.pkl', 'rb') as f:
        return pickle.load(f)


def save(data:np.ndarray = None,path:str = None) -> None:
    np.save(path + '.npy', data, allow_pickle=True)


def load(path:str = None) -> np.ndarray:
    return np.load(path + '.npy', allow_pickle=True) 

# %% [markdown]
# # Data Reading & Injection

# %%
current_dir = os.getcwd()
data_dir = os.path.join('../data', 'fma_metadata')
result_dir = os.path.join(current_dir, 'results')

tracks = pd.read_csv(
    os.path.join(data_dir, 'tracks.csv'),
    index_col=0, 
    header = [0, 1]
)

features = pd.read_csv(
    os.path.join(data_dir, "features.csv"),
    index_col=0,
    header = [0, 1, 2]
)

# %% [markdown]
# # Data Splitting

# %%
small = tracks['set', 'subset'] <= 'small'

train = tracks['set', 'split'] == 'training'
val = tracks['set', 'split'] == 'validation'
test = tracks['set', 'split'] == 'test'

y_train = tracks.loc[small & train, ('track', 'genre_top')]
y_test = tracks.loc[small & test, ('track', 'genre_top')]
X_train = features.loc[small & train, 'mfcc']
X_test = features.loc[small & test, 'mfcc']

# %%
X_train.shape, y_train.shape, X_test.shape, y_test.shape

# %%
feature_cols = X_train.columns

# %%
undrop_cond_train = y_train.isna() != True
y_train = y_train[undrop_cond_train]
X_train = X_train[undrop_cond_train]

undrop_cond_test = y_test.isna() != True
y_test = y_test[undrop_cond_test]
X_test = X_test[undrop_cond_test]

# %% [markdown]
# # Dimension Reduction & Visualization

# %%
import plotly.io as plt_io
import plotly.graph_objects as go


def plot_2d(component1:np.ndarray, component2:np.ndarray,  path:str, y = None,) -> None:
    
    fig = go.Figure(data=go.Scatter(
        x = component1,
        y = component2,
        mode='markers',
        marker=dict(
            size=20,
            color=y, #set color equal to a variable
            colorscale='Rainbow', # one of plotly colorscales
            showscale=True,
            line_width=1
        )
    ))
    fig.update_layout(margin=dict(l=100,r=100,b=100,t=100),width=2000,height=1200)                 
    fig.layout.template = 'plotly_dark'
    
    fig.show()
    
    
    fig.write_image(path)

def plot_3d(component1: np.ndarray,
            component2 : np.ndarray,
            component3 :np.ndarray,
            path:str,
            y = None) -> None:
    
    fig = go.Figure(data=[go.Scatter3d(
            x=component1,
            y=component2,
            z=component3,
            mode='markers',
            marker=dict(
                size=10,
                color=y,                # set color to an array/list of desired values
                colorscale='Rainbow',   # choose a colorscale
                opacity=1,
                line_width=1
            )
        )])
    # tight layout
    fig.update_layout(margin=dict(l=50,r=50,b=50,t=50),width=1800,height=1000)
    fig.layout.template = 'plotly_dark'

    fig.show()
    fig.write_image(path)
    

# %%
from sklearn.preprocessing import LabelEncoder
y = LabelEncoder().fit_transform(y_train)

# %%
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# %% [markdown]
# ## PCA

# %%
x = X_train.copy()
pca = PCA(n_components=3)
principalComponents = pca.fit_transform(x)

principal = pd.DataFrame(
    data = principalComponents,
    columns = [
        'PC 1',
        'PC 2',
        'PC 3'
    ]
)

plot_2d(
    principalComponents[:, 0],
    principalComponents[:, 1],
    y = y,
    path = os.path.join(current_dir, result_dir, 'pca_2d.png')
)

# %%
plot_3d(
    principalComponents[:, 0],
    principalComponents[:, 1],
    principalComponents[:, 2],
    path = os.path.join(current_dir, result_dir, 'pca_3d.png'),
    y = y
)

# %% [markdown]
# ## Linear Discriminant Analysis

# %%
x = X_train.copy()
lda = LDA(n_components=3)
embedding = lda.fit_transform(x, y)

plot_3d(
    embedding[:, 0],
    embedding[:, 1],
    embedding[:, 2],
    path = os.path.join(current_dir, result_dir, 'lda_3d.png'),
    y = y
)

# %% [markdown]
# ## Manifold Learning : T-distributed Stochastic Neighbor Embedding

# %%
x = X_train.copy()
pca = PCA(n_components=100)
principalComponents = pca.fit_transform(x)

tsne = TSNE(n_components=3)
embedding = tsne.fit_transform(principalComponents)

plot_3d(
    embedding[:, 0],
    embedding[:, 1],
    embedding[:, 2],
    path = os.path.join(current_dir, result_dir, 'TSNE_3d.png'),
    y = y
)

# %% [markdown]
# # Modelling 

# %%
unique_genres = np.unique(y_train)
#unique_genres

# %%
from sklearn.ensemble import (
    RandomForestClassifier,
    VotingClassifier,
    BaggingClassifier,
    ExtraTreesClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
)

from sklearn.linear_model import (
    LogisticRegression, 
    Perceptron,
    SGDClassifier,
    RidgeClassifier,
)

from sklearn.svm import (
    SVC, 
    LinearSVC,
    NuSVC
)


from sklearn.neighbors import (
    KNeighborsClassifier,
    NearestCentroid
)

from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

# %%
models = {
    RandomForestClassifier.__name__ : RandomForestClassifier(),
    BaggingClassifier.__name__: BaggingClassifier(),
    ExtraTreesClassifier.__name__ :ExtraTreesClassifier(),
    AdaBoostClassifier.__name__: AdaBoostClassifier(),
    #GradientBoostingClassifier.__name__: GradientBoostingClassifier(),
    LogisticRegression.__name__: LogisticRegression(),
    Perceptron.__name__: Perceptron(),
    SGDClassifier.__name__: SGDClassifier(),
    RidgeClassifier.__name__: RidgeClassifier(),
    SVC.__name__: SVC(),
    LinearSVC.__name__: LinearSVC(),
    #NuSVC.__name__: NuSVC(),
    KNeighborsClassifier.__name__: KNeighborsClassifier(),
    NearestCentroid.__name__: NearestCentroid(),
    MLPClassifier.__name__: MLPClassifier(),
    DecisionTreeClassifier.__name__: DecisionTreeClassifier(),
    GaussianNB.__name__: GaussianNB(),
}

metrics = {
    accuracy_score.__name__ : accuracy_score, 
    recall_score.__name__ : recall_score, 
    precision_score.__name__: precision_score,
    f1_score.__name__ : f1_score, 
    #roc_auc_score.__name__ : roc_auc_score
}

# %% [markdown]
# ## Data Scaling

# %%
scaler = StandardScaler() #or MinMax
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# %% [markdown]
# ## Model Selection Loop

# %%
scores = {}

for model_name, model in models.items():
    print(f"{model_name} is fitting." )

    model_cache = {}
    since = time.time()

    model.fit(
        X_train,
        y_train
    )

    model_cache['fitted_time(s)'] = time.time() - since

    preds = model.predict(X_test)

    
    for metric_name, metric in metrics.items():

        if any(
            [
                metric_name == 'precision_score',
                metric_name == 'recall_score',
                metric_name == 'f1_score'
            ]
        ):
            model_cache[metric_name] = metric(
                y_test,
                preds,
                average='micro'
            )

        else:
            model_cache[metric_name] = metric(
                y_test,
                preds
            )


    scores[model_name] = model_cache

# %%


# %%
df_scores = pd.DataFrame(scores).T

df_scores = df_scores[
    [
       'accuracy_score',
       'precision_score',
       'recall_score',
       'f1_score',
       'fitted_time(s)'
    ]
].sort_values('accuracy_score', ascending = False)

# %%
df_scores.to_csv(
    os.path.join(result_dir, 'ml-baseline-scores.csv')
)

# %% [markdown]
# # Cross-Validation

# %%
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report

# %%
scores = cross_val_score(
    MLPClassifier(
        hidden_layer_sizes= (64, ),
        max_iter=200,
        warm_start=True,
    ),
    X_train,
    y_train, 
    cv=5,
    #return_estimator=True
)

# %%
print(" Multi-layer Percoptron Classifier has %0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))

# %%
scores = cross_val_score(
    SVC(
        kernel = 'rbf'
    ),
    X_train, 
    y_train,
    cv=5
)

print(" Support Vector Classifier has %0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))

# %% [markdown]
# # Ensemble Learning

# %%
ensemble_models = [
    #GradientBoostingClassifier(),
    VotingClassifier(
        estimators = [
            ('MLP', MLPClassifier()),
            ('SVC', SVC(kernel = 'rbf'))
        ]
    ),
    VotingClassifier(
        estimators = [
            ('KNN', KNeighborsClassifier()),
            ('RFC', RandomForestClassifier(n_estimators=50, random_state=1)),
            ('GaussianNB', GaussianNB())
        ]
    ),
    VotingClassifier(
        estimators = [
            ('MLP', MLPClassifier()),
            ('RFC', RandomForestClassifier(n_estimators=50, random_state=1)),
            ('SVC', SVC(kernel = 'rbf'))
        ]
    )
]


scores = {}

for model in ensemble_models:
    print(f"{model} is fitting." )

    model_cache = {}
    since = time.time()

    model.fit(
        X_train,
        y_train
    )

    model_cache['fitted_time(s)'] = time.time() - since

    preds = model.predict(X_test)

    
    for metric_name, metric in metrics.items():

        if any(
            [
                metric_name == 'precision_score',
                metric_name == 'recall_score',
                metric_name == 'f1_score'
            ]
        ):
            model_cache[metric_name] = metric(
                y_test,
                preds,
                average='micro'
            )

        else:
            model_cache[metric_name] = metric(
                y_test,
                preds
            )


    scores[str(model)] = model_cache


# %%
df_scores = pd.DataFrame(scores).T

df_scores = df_scores[
    [
       'accuracy_score',
       'precision_score',
       'recall_score',
       'f1_score',
       'fitted_time(s)'
    ]
].sort_values('accuracy_score', ascending = False)
#df_scores

# %%
df_scores.to_csv(
    os.path.join(result_dir, 'ml-ensemble-baseline-scores.csv')
)

# %% [markdown]
# # Confusion Matrix of Best Model

# %%
preds = MLPClassifier(
        hidden_layer_sizes= (64, ),
        max_iter=400,
        warm_start=True,
    ).fit(
        X_train,
        y_train
    ).predict(
        X_test
)

conf_matrix = confusion_matrix(
        y_test.values,
        preds
    )
    
#conf_matrix

# %%
conf_matrix.to_csv(
    os.path.join(current_dir, result_dir, 'conf_mat.csv')
)

# %%
visualize_confusion_matrix(conf_matrix)

# %% [markdown]
# # Feature Importance

# %% [markdown]
# ## Random Forest Classifier

# %%
from sklearn.ensemble import (
    RandomForestClassifier
)
forest = RandomForestClassifier(random_state=0)
forest.fit(X_train, y_train)
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)

# %%
sorted_idx = importances.argsort()


plt.figure(figsize = (18, 9))
plt.barh(
    list(
    map(
        lambda x: ' '.join(x),
        feature_cols[sorted_idx][:50]
    )
), 
importances[sorted_idx][:50],
color = 'orange'
)

# %%
forest_importances = pd.Series(importances[sorted_idx][:50], index=list(
    map(
        lambda x: ' '.join(x),
        feature_cols[sorted_idx][:50]
    )
)
)



fig, ax = plt.subplots(figsize=(18, 6))
forest_importances.plot.bar(yerr=std[sorted_idx][:50], ax=ax)
ax.set_title("Feature importances using MDI")
ax.set_ylabel("Mean decrease in impurity")
fig.tight_layout()

# %%



