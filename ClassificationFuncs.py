import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score



def train_test_preparator(cleaned_dataset, numeric_cols, col_label, min_freq = 15):
    """This function prepares the training and test data for the classification.
    It removes data samples with labels that have low frequency.It then transforms 
    features and label sets into numpy arrays and then standardizes them.
    
    Parameters
    ----------
    cleaned_dataset : pandas.DataFrame
        Cleaned data
    numeric_cols : list
        List of columns to be used as features
    col_label : str
        Label column
    min_freq : int
        Minimum frequency of a class (label)
    Returns
    ----------
    numpy.ndarray
    numpy.ndarray
    numpy.ndarray
    numpy.ndarray
    """
    
    # Remove data samples with labels that have frequency lower than min_freq
    value_counts = cleaned_dataset[col_label].value_counts()
    to_remove = value_counts[value_counts <= min_freq].index
    cleaned_dataset[col_label].replace(to_remove, np.nan, inplace=True)
    cleaned_dataset = cleaned_dataset[cleaned_dataset[col_label].notna()]
    
    # Prepare features and labels arrays
    X = cleaned_dataset[numeric_cols].values
    Y = cleaned_dataset[col_label].values
    Y = LabelEncoder().fit_transform(Y)
    
    # Create training and test data sets
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, 
                                                        random_state=0)
    
    # Standardize data
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test


def data_classifier(X_train, X_test, y_train, y_test, estimator_type=None, 
                    performance_measure='f1_macro'):
    """Classifier
    This function trains an estimator model on the training data, validates them,
    and then measures the performance on the test set.
    
    
    Parameters
    ----------
    X_train : numpy.ndarray
        Training set features (input)
    X_test : numpy.ndarray
        Test set features (input)
    y_train : numpy.ndarray
        Training set labels (output)
    y_test : numpy.ndarray
        Test set labels (output)
    estimator_type : str
        The estimator to use
    performance_measure : str
        Classification performance measure
    Returns
    ----------
    classification_performance
        float
    """
    if estimator_type == 'Logistic Regression':
        param_grid = {'C': [0.01, 0.1, 0.5, 0.7, 1, 2, 3, 3.5, 4, 6, 8, 10, 12, 100, 1000]}
        clf = GridSearchCV(LogisticRegression(penalty='l2', class_weight='balanced', 
                                          solver='lbfgs', multi_class='ovr', max_iter=1000, 
                                          dual=False), param_grid, cv=10, scoring=performance_measure)
    elif estimator_type == 'k-nearest neighbours':
        clf = KNeighborsClassifier(n_neighbors = 7)
    elif estimator_type == 'Random Forest':
        clf = RandomForestClassifier(n_estimators=100, max_features="sqrt", max_depth=None, 
                             min_samples_split=2, oob_score=True, n_jobs=-1)
    elif estimator_type == 'Linear SVM':
        param_grid = {'C': [0.001, 0.01, 0.1, 0.5, 0.7, 1, 2, 3, 3.5, 4, 6, 8, 10, 12, 
                        50, 100, 500, 1000]}
        clf = GridSearchCV(LinearSVC(loss='squared_hinge', class_weight='balanced', 
                                     max_iter=10000, dual=False), param_grid, cv=10, scoring=performance_measure)
    elif estimator_type == 'AdaBoost':
        clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=3, min_samples_leaf=1),
                         algorithm="SAMME.R", n_estimators=200, learning_rate=1)
    else:
        param_grid = {'C': [0.001, 0.01, 0.1, 0.5, 0.7, 1, 2, 3, 3.5, 4, 6, 8, 10, 12, 
                        50, 100, 500, 1000], 'gamma': [10, 1, 0.1, 0.05, 0.01, 0.005, 
                                          0.007, 0.001, 0.0005, 0.0001, 0.00001, 0.000001], 'kernel': ['rbf']}
        clf = GridSearchCV(SVC(decision_function_shape='ovo', class_weight='balanced'), 
                       param_grid, cv=10, scoring=performance_measure)
    
    clf_train = clf.fit(X_train, y_train)
    clf_predictions = clf_train.predict(X_test)
    
    if performance_measure == 'accuracy':
        classification_performance = accuracy_score(y_test, clf_predictions)
    elif performance_measure == 'f1_macro':
        classification_performance = f1_score(y_test, clf_predictions, average='macro')
    else:
        clf_proba = clf_train.predict_proba(X_test)
        classification_performance = roc_auc_score(y_test, clf_proba, multi_class='ovo', average='macro')
    #TODO: Adjust hyperparameters for the estimators using GridsearchCV
    return classification_performance
