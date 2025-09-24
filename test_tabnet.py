from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np  
from sklearn.utils._tags import InputTags
class SklearnCompatibleTabNetClassifier(BaseEstimator, ClassifierMixin):
    _estimator_type = "classifier"  # Tells scikit-learn it's a classifier

    def __init__(self, **kwargs):
        self.tabnet = TabNetClassifier(**kwargs)

    def __sklearn_tags__(self):
        class DummyTag:
            def __init__(self):
                self.estimator_type = 'classifier'
                self.input_tags=InputTags(one_d_array=False, two_d_array=True, three_d_array=False, sparse=True, categorical=False, string=False, dict=False, positive_only=False, allow_nan=False, pairwise=False)
                self.requires_fit=True
        return DummyTag()

    def fit(self, X, y, **fit_params):
        print("model.fit called")
        print(fit_params)
        self.tabnet.fit(X, y, **fit_params)
        self.classes_ = np.unique(y)  # required by scikit-learn
        return self

    def predict(self, X):
        return self.tabnet.predict(X)

    def predict_proba(self, X):
        return self.tabnet.predict_proba(X)

    def score(self, X, y):
        from sklearn.metrics import accuracy_score
        return accuracy_score(y, self.predict(X))
    
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Make example data (or use your own X_train, y_train)
X, y = make_classification(n_samples=300, n_features=20, random_state=42)
X = X.astype('float32')  # required by TabNet
y = y.astype(int)        # must be int (not float) for classification

model = TabNetClassifier(
    n_d=8, n_a=8, n_steps=3, verbose=1,
)
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)
print("Tabnet classifier fit:")
model.fit(
    X_train, y_train,
    batch_size=32, 
    max_epochs=5,
    virtual_batch_size=16,    
)
print("Tabnet classifier done.")

model2 = SklearnCompatibleTabNetClassifier(
    n_d=8, n_a=8, n_steps=3, verbose=1,
)
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

print("SklearnCompatibleTabNetClassifier fit:")
model2.fit(
    X_train, y_train,
    batch_size=32, 
    virtual_batch_size=16,    
    max_epochs=5,
)
print("SklearnCompatibleTabNetClassifier done.")



from sklearn.model_selection import cross_val_score
# print("Cross-validation AUC scores:")
scores = cross_val_score(model2, X, y, scoring='roc_auc', cv=3, params={
    'batch_size': 32,
    'virtual_batch_size': 16,
    'max_epochs': 5,
})
print("Scores:", scores)