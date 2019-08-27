"""
Implements the book chapter 7 on Cross Validation for financial data.
"""

import pandas as pd
import numpy as np
from scipy.stats import rv_continuous, kstest

from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import log_loss, accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import KFold, GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline

import matplotlib.pyplot as plt


def ml_get_train_times(info_sets: pd.Series, test_times: pd.Series) -> pd.Series:
    # pylint: disable=invalid-name
    """
    Snippet 7.1, page 106,  Purging observations in the training set

    This function find the training set indexes given the information on which each record is based
    and the range for the test set.

    Given test_times, find the times of the training observations.
    :param info_sets: The information on which each record is constructed from
        -info_sets.index: Time when the information extraction started.
        -info_sets.value: Time when the information extraction ended.
    :param test_times: Times for the test dataset.
    """
    train = info_sets.copy(deep=True)
    for start_ix, end_ix in test_times.iteritems():
        df0 = train[(start_ix <= train.index) & (train.index <= end_ix)].index  # Train starts within test
        df1 = train[(start_ix <= train) & (train <= end_ix)].index  # Train ends within test
        df2 = train[(train.index <= start_ix) & (end_ix <= train)].index  # Train envelops test
        train = train.drop(df0.union(df1).union(df2))
    return train


class PurgedKFold(KFold):
    """
    Snippet 7.3, page 109, Cross-Validation Class when Observations Overlap.

    Extend KFold class to work with labels that span intervals.
    The train is purged of observations overlapping test-label intervals.
    Test set is assumed contiguous (shuffle=False), w/o training samples in between.
    """

    def __init__(self, info_sets, n_splits=3, pct_embargo=0., random_state=None):
        """
        Constructor

        :param info_sets (pd.Series):
            —info_sets.index: Time when the information extraction started.
            —info_sets.value: Time when the information extraction ended.
        :param n_splits: The number of splits. Default to 3
        :param pct_embargo: Percent that determines the embargo size.
        :param random_state: (int or RandomState): random state
        """
        if not isinstance(info_sets, pd.Series):
            raise ValueError('The info_sets param must be a pd.Series')
        super(PurgedKFold, self).__init__(n_splits, shuffle=False, random_state=random_state)

        # Attributes
        self.info_sets = info_sets
        self.pct_embargo = pct_embargo

    # noinspection PyPep8Naming
    def split(self, X, y=None, groups=None):
        if X.shape[0] != self.info_sets.shape[0]:
            raise ValueError("X and the 'info_sets' series param must be the same length")

        indices: np.ndarray = np.arange(X.shape[0])
        embargo: int = int(X.shape[0] * self.pct_embargo)

        test_ranges: [(int, int)] = [(ix[0], ix[-1] + 1) for ix in np.array_split(np.arange(X.shape[0]), self.n_splits)]
        for start_ix, end_ix in test_ranges:
            test_indices = indices[start_ix:end_ix]

            if end_ix < X.shape[0]:
                end_ix += embargo

            test_times = pd.Series(index=[self.info_sets[start_ix]], data=[self.info_sets[end_ix - 1]])
            train_times = ml_get_train_times(self.info_sets, test_times)

            train_indices = []
            for train_ix in train_times.index:
                train_indices.append(self.info_sets.index.get_loc(train_ix))
            yield np.array(train_indices), test_indices


# noinspection PyPep8Naming
def ml_cross_val_score(classifier, X, y, cv_gen, sample_weight=None, scoring='neg_log_loss'):
    # pylint: disable=invalid-name
    """
    Snippet 7.4, page 110, Using the PurgedKFold Class.
    Function to run a cross-validation evaluation of the using sample weights and a custom CV generator.

    Note: This function is different to the book in that it requires the user to pass through a CV object. The book
    will accept a None value as a default and then resort to using PurgedCV, this also meant that extra arguments had to
    be passed to the function. To correct this we have removed the default and require the user to pass a CV object to
    the function.

    Example:
    cv_gen = PurgedKFold(n_splits=n_splits, info_sets=info_sets, pct_embargo=pct_embargo)
    scores_array = ml_cross_val_score(classifier, X, y, cv_gen, sample_weight=None, scoring='neg_log_loss')

    :param classifier: A sk-learn Classifier object instance.
    :param X: The dataset of records to evaluate.
    :param y: The labels corresponding to the X dataset.
    :param cv_gen: Cross Validation generator object instance.
    :param sample_weight: A numpy array of weights for each record in the dataset.
    :param scoring: A metric name to use for scoring; currently supports `neg_log_loss`, `accuracy`, `f1`, `precision`,
        `recall`, and `roc_auc`.
    :return: The computed score as a numpy array.
    """
    # Define scoring metrics
    scoring_func_dict = {'neg_log_loss': log_loss, 'accuracy': accuracy_score, 'f1': f1_score,
                         'precision': precision_score, 'recall': recall_score, 'roc_auc': roc_auc_score}
    try:
        scoring_func = scoring_func_dict[scoring]
    except KeyError:
        raise ValueError('Wrong scoring method. Select from: neg_log_loss, accuracy, f1, precision, recall, roc_auc')

    # If no sample_weight then broadcast a value of 1 to all samples (full weight).
    if sample_weight is None:
        sample_weight = np.ones((X.shape[0],))

    # Score model on KFolds
    ret_scores = []
    for train, test in cv_gen.split(X=X, y=y):
        fit = classifier.fit(X=X.iloc[train, :], y=y.iloc[train], sample_weight=sample_weight[train])
        if scoring == 'neg_log_loss':
            prob = fit.predict_proba(X.iloc[test, :])
            score = -1 * scoring_func(y.iloc[test], prob, sample_weight=sample_weight[test], labels=classifier.classes_)
        else:
            pred = fit.predict(X.iloc[test, :])
            score = scoring_func(y.iloc[test], pred, sample_weight=sample_weight[test])
        ret_scores.append(score)
    return np.array(ret_scores)


def clf_hyper_fit(feat, lbl, t1, pipe_clf, param_grid, cv=3, bagging=(0.0, None, 1.0), n_jobs=-1, random_iterations=0,
                  pct_embargo=0.0, **fit_params):
    """
    Snippet 9.1 & 9.3, page 130 & 132, Grid Search & Randomized Search with Purged K-Fold Cross-Validation.

    """
    if set(lbl.values) == {0, 1}:
        scoring = 'f1'  # F1 for meta labelling
    else:
        scoring = 'neg_log_loss'  # Symmetric towards all cases

    # Step 1) Hyperparameter search, on train data
    inner_cv = PurgedKFold(n_splits=cv, info_sets=t1, pct_embargo=pct_embargo)  # Purged

    if random_iterations == 0:
        gs = GridSearchCV(estimator=pipe_clf, param_grid=param_grid, scoring=scoring, cv=inner_cv, n_jobs=n_jobs,
                          iid=False)
    else:
        gs = RandomizedSearchCV(estimator=pipe_clf, param_distributions=param_grid, scoring=scoring, cv=inner_cv,
                                n_jobs=n_jobs, iid=False, n_iter=random_iterations)
    gs = gs.fit(feat, lbl, **fit_params).best_estimator_  # Pipeline

    # Step 2.) Fit validated model on the entirety of the data
    if bagging[1] > 0:
        gs = BaggingClassifier(base_estimator=MyPipeline(gs.steps), n_estimators=int(bagging[0]),
                               max_samples=float(bagging[1]), max_features=float(bagging[2]), n_jobs=n_jobs)
        gs = gs.fit(feat, lbl, sample_weight=fit_params[gs.base_estimator.steps[-1][0]+'__sample_weight'])
        gs = Pipeline([('bag', gs)])

    return gs


class MyPipeline(Pipeline):
    """
    Snippet 9.2, page 131, An Enhanced Pipeline Class

    This example introduces nicely one limitation of sklearn’s Pipelines : Their fit method does not expect a
    sample_weight argument. Instead, it expects a fit_params keyworded argument. That is a bug that has been reported
    in GitHub; however, it may take some time to fix it, as it involves rewriting and testing much functionality.

    Until then, feel free to use the workaround in Snippet 9.2. It creates a new class, called MyPipeline, which
    inherits all methods from sklearn’s Pipeline. It overwrites the inherited fit method with a new one that handles
    the argument sample_weight, after which it redirects to the parent class.

    If you are not familiar with this technique for expanding classes, you may want to read this introductory
    Stackoverflow post: http://stackoverflow.com/questions/ 576169/understanding-python-super-with-init-methods.
    """

    def fit(self, X, y, sample_weight=None, **fit_params):
        if sample_weight is not None:
            fit_params[self.steps[-1][0] + '__sample_weight'] = sample_weight

        return super(MyPipeline, self).fit(X, y, **fit_params)


# ============================================
# Add the loguniform distribution

class LogUniformGenerator(rv_continuous):
    # Random numbers log-uniformly distributed between 1 and e
    def _cdf(self, x):
        return np.log(x/self.a) / np.log(self.b/self.a)


def log_uniform(a=1.0, b=np.exp(1)):
    return LogUniformGenerator(a=a, b=b, name='logUniform')


if __name__ == '__main__':
    # Code regarding log_uniform
    a, b, size = 1E-3, 1E3, 10000
    vals = log_uniform(a=a, b=b).rvs(size=size)
    print(kstest(rvs=np.log(vals), cdf='uniform', args=(np.log(a), np.log(b / a)), N=size))
    print(pd.Series(vals).describe())
    plt.subplot(121)
    pd.Series(np.log(vals)).hist()
    plt.subplot(122)
    pd.Series(vals).hist()
    plt.show()