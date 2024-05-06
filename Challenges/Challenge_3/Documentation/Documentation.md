## Web Scraping

To run the webscrapping you have to follow the nexst steps:  

#### 1. Import required libraries

```shell
import googleapiclient.discovery
  
```
Since we are working with a googleapi is necesary for you to get a key, you cand do so by going to goole cloud and downloading [YouTube API](https://developers.google.com/youtube/v3/getting-started)

update dev with your key and you are ready to run the code.

```shell
dev = "your_api"  
```
This will create a df with the info and will be saved as a csv file, you can update that

```shell
csv_file_path = 'comments_data.csv'
df.to_csv(csv_file_path, index=False)  
```

#### 1. Import required libraries

## Model construction

```shell
import googleapiclient.discovery
  
```

This repository contains _educational_ reference material to illustrate how to webscrap and do a sentiment analysis.

## MLOps

1. Install the required libraries:
```shell
$ pip install pandas==2.0.3
$ pip install nltk==3.8.1
$ pip install numpy==1.24.3
$ pip install mlflow==2.11.1
$ pip install google-api-python-client
$ pip install langdetect
$ pip install scikit-learn==1.2.2
  
```
</details> 

For more details check de documentation folder.
