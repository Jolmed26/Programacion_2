## Model construction

To construct the model you would have to follow next steps once you worked your df and have it ready to work with:

#### 1. Import required libraries
You can check the version of each library on the readme file in the challenge_2 folder.

```shell
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, precision_recall_curve, auc
```
#### 2. Define the metrics

A confusion matrix is a table that categorizes predictions based on whether they match the actual values.

- True Positive (TP): Turnover correctly classified as Turnover.
- True Negative (TN): Not turnover correctly classified as Not turnover.
- False Positive (FP): Not turnover incorrectly classified as turnover.
- False Negative (FN): turnover incorrectly classified as Not turnover.

Metrics:

- Accuracy (also known as success rate): (TP+TN)/(TP+TN+FP+FN)
- Sensitivity (also known as true positive rate): TP/(TP+FN)
- Specificity (also known as true negative rate): TN/(TN+FP)
- Precision (also known as positive predictive value): TP/(TP+FP)
- Recall: same as sensitivity
- F-measure: 2PrecisionRecall/(Recall+Precision) = 2TP/(2TP+FP+FN)

```shell
def metrics(X,CV_clf):
    y_pred = CV_clf.predict(X)
    cm = confusion_matrix(y_test, y_pred)
    tn = cm[0,0]
    fp = cm[0,1]
    fn = cm[1,0]
    tp = cm[1,1]
    Accuracy=(tp+tn)/(tp+tn+fp+fn)
    Sensitivity=tp/(tp+fn)
    Specificity=tn/(tn+fp)
    Precision=tp/(tp+fp)
    F_measure=2*tp/(2*tp+fp+fn)
    print('Accuracy=%.3f'%Accuracy)
    print('Sensitivity=%.3f'%Sensitivity) # como recall
    print('Specificity=%.3f'%Specificity)
    print('Precision=%.3f'%Precision)
    print('F-measure=%.3f'%F_measure)
    return Accuracy, Sensitivity, Specificity, Precision, F_measure
```

#### 3. Get the clasifier set and get your data split

First you have to find the best parameters, to do so you have to use:

```shell
classifier_log = LogisticRegression(random_state=random_state,solver='lbfgs', max_iter=1000)
parameters_log = {
            'penalty' : ['l2'],  
            'C' : [0.01, 0.1, 1, 10, 100]
}
scoring='accuracy'   # scoring parameters: https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
# Encontrar los mejores hiperparámetros
cv_results, best_param, best_result = modelselection(classifier_log,parameters_log, scoring, X_train)
```


#### 4. Evaluate your model

To evaluate your model you will have Accuracy, sensitivity, specificity, precision and F-measure

We can apply it to the metrics already defined:

```shell
# Clasificador con los mejores hyperparameters
logReg_clf = LogisticRegression(penalty = best_param['penalty'],
                            C = best_param['C'],
                            random_state=random_state)
logReg_clf.fit(X_train, y_train)

# Metrics
logReg_metrics = metrics(X_test,logReg_clf)
```


## MLOps

#### 1. Import required libraries

```shell
import mlflow
import mlflow.sklearn
```

#### 2. Get the MLflow tracking ready

To get your mlflow set you have to use next parameters

```shell
mlflow.set_tracking_uri('http://127.0.0.1:5000')
mlflow.set_experiment('BreastCancer')
```

Make sure of update the tracking_uri to your URL, this is running locally, to get your own url you have to go to your console and run `mlflow ui`.

Also update the experiment name `set_experiment` to the one needed.

#### 3. Start the MLflow run and log parameters 

To evaluate your model you will have 2 elements, fist one is the confusion matrix:

```shell
# Start MLflow run and log parameters
with mlflow.start_run():
    mlflow.log_param('Logistic_regression', 1)

    # Log evaluation metrics
    mlflow.log_metric('Accuracy', logReg_metrics[0])
    mlflow.log_metric('Sensitivity', logReg_metrics[1])
    mlflow.log_metric('Specificity', logReg_metrics[2])
    mlflow.log_metric('Precision', logReg_metrics[3])
    mlflow.log_metric('F-measure', logReg_metrics[4])

    # Log confusion matrix as artifact
    mlflow.log_artifact('roc.png', 'roc.png')

    # Log model performance plot as artifact
    mlflow.log_artifact('model_performance.png', 'model_performance.png')  # Log the PNG file

print('MLflow run completed.')
```
finally you have to update the `nb_score` for the value needed as the score.
