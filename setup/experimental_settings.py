# scalers methods
from sklearn import svm
from sklearn.covariance import EllipticEnvelope
from sklearn.kernel_approximation import Nystroem
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer

# regression methods
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge, SGDOneClassSVM
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, IsolationForest
from sklearn.neighbors import KNeighborsRegressor, LocalOutlierFactor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

# classification methods
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier


def get_regressors():
    regressors = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Lasso Regression': Lasso(alpha=0.1),
        'Elastic Net': ElasticNet(alpha=0.1, l1_ratio=0.5),
        'Bayesian Ridge': BayesianRidge(),
        # 'ARDRegression': ARDRegression(),
        # 'SGDRegressor': SGDRegressor(),
        'Decision Tree': DecisionTreeRegressor(),
        'Random Forest': RandomForestRegressor(n_estimators=100),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100),
        'AdaBoost Regressor': AdaBoostRegressor(n_estimators=100),
        'KNeighbors Regressor': KNeighborsRegressor(),
        'Support Vector Regressor (SVR)': SVR(),
        'Multi-layer Perceptron (MLP)': MLPRegressor()
    }

    return regressors


def get_classifiers():
    classifiers = {
        # 'Logistic Regression': LogisticRegression(),
        'K-Nearest Neighbors': KNeighborsClassifier(),
        'Decision Tree': DecisionTreeClassifier(),
        'Random Forest': RandomForestClassifier(),
        # 'Gaussian Naive Bayes': GaussianNB(),
        'Support Vector Machine': SVC(),
        'XGBoost': XGBClassifier(use_label_encoder=True),
        # 'MLPClassifier': MLPClassifier()
    }

    return classifiers


def get_anomaly_detectors(outliers_fraction):
    classifiers = {
        # 'Robust covariance': EllipticEnvelope(contamination=outliers_fraction),
        'One-Class SVM': svm.OneClassSVM(nu=outliers_fraction, kernel="rbf", gamma=0.1),
        #'One-Class SVM (SGD)': make_pipeline(
        #    Nystroem(gamma=0.1, random_state=42, n_components=1),
        #    SGDOneClassSVM(
        #        nu=outliers_fraction,
        #        shuffle=True,
        #        fit_intercept=True,
        #        random_state=42,
        #        tol=1e-6,
        #    ),
        #),
        'Isolation Forest': IsolationForest(contamination=outliers_fraction, random_state=42)
        #'Local Outlier Factor': LocalOutlierFactor(contamination=outliers_fraction)
    }

    return classifiers


def get_scalers():
    return [MinMaxScaler(), StandardScaler(), Normalizer(), None]
