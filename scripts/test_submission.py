"""
Simple pipeline to test the naive submission.
"""

import numpy as np
import imp

from sklearn.pipeline import Pipeline

import problem


# Import corpus ------------------------------------------------------------------------

X_train_full, y_train_full = problem.get_train_data(sep='|')
cv_list = list(problem.get_cv(X_train_full, y_train_full))

fold = np.random.randint(0, 6)
print('fold: ' + str(fold))

X_train = X_train_full.iloc[cv_list[fold][0], :]
X_test = X_train_full.iloc[cv_list[fold][1], :]
y_train = y_train_full[cv_list[fold][0]]
y_test = y_train_full[cv_list[fold][1]]


# Test submission on 1 fold ------------------------------------------------------------

scorer = problem.F1Score()

submission_name = 'starting_kit'

extractor_module = imp.load_source('', 'submissions/' + submission_name + '/feature_extractor.py')
feature_extractor = extractor_module.FeatureExtractor()

classifier_module = imp.load_source('', 'submissions/' + submission_name + '/classifier.py')
classifier = classifier_module.Classifier()

pipeline = Pipeline([
    ('vectorizer', feature_extractor),
    ('clf', classifier),
])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

scorer(y_true=y_test, y_pred=y_pred)
