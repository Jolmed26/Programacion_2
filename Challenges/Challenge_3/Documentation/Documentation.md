## Web Scraping

To run the webscrapping you have to follow the nexst steps:  

#### 1. Import required libraries

```shell
import googleapiclient.discovery
```

#### 2. Get your API key

Since we are working with a googleapi is necesary for you to get a key, you cand do so by going to goole cloud and downloading [YouTube API](https://developers.google.com/youtube/v3/getting-started)

update dev with your key and you are ready to run the code.

```shell
dev = "your_api"  
```

#### 3. Create a df with info scraped

This will create a df with the info and will be saved as a csv file, you can update that

```shell
csv_file_path = 'comments_data.csv'
df.to_csv(csv_file_path, index=False)  
```


## Model construction

To construct the model you would have to follow next steps once you worked your df and have it ready to work with:

#### 1. Import required libraries

```shell
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
from sklearn.feature_extraction.text import CountVectorizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
```

#### 2. Get the clasifier set and get your data split

This is quite simple, you have to ser your classifier, in this model we use GaussianNB.

```shell
# Deffine the clasifier 
classifier = GaussianNB()

# Spliting data into test and training
X_trainn, X_testn, y_trainn, y_testn = train_test_split(Xn, yn, test_size=0.3, random_state=0)
classifier.fit(X_trainn, y_trainn)
```
Make sure of update your `test_size` to what you need for your model, in this case, this values gave extraordinary results.

#### 3. Evaluate your model

To evaluate your model you will have 2 elements, fist one is the confusion matrix:

```shell
y_predn = classifier.predict(X_testn)
cmn = confusion_matrix(y_testn, y_predn)

# Save the confusion matrix as a file
np.save('confusion_matrixen.npy', cmn)
```
then you will have the accuracy one:

```shell
nb_scoren = accuracy_score(y_testn, y_predn)
print('accuracy score for english prediction',nb_scoren)
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
mlflow.set_experiment('NLP_EN_PD')
```

Make sure of update the tracking_uri to your URL, this is running locally, to get your own url you have to go to your console and run `mlflow ui`.

Also update the experiment name to the one needed.

#### 3. Start the MLflow run and log parameters 

To evaluate your model you will have 2 elements, fist one is the confusion matrix:

```shell
with mlflow.start_run():
    mlflow.log_param('normilizedat', 1027)

    # Log evaluation metrics
    mlflow.log_metric('accuracy', nb_scores)
    
    # Log confusion matrix as artifact
    mlflow.log_artifact('confusion_matrixes.npy', 'confusion_matrixes.npy')

print('MLflow run completed.')
```
finally you have to update the `nb_score` for the value needed as the score.
