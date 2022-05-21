# [APEX &middot; DASH]

## Introduction

## Instructions for deployment

### Sentiment Analysis

Two things are required to run with the Sentiment Analysis feature.

1. Twitter API tokens. a `sample_twitter.csv` file is provided, where the fields need to be filled with the corresponding Twitter keys. The file needs to be renamed to `twitter.csv`.

2. In line 41 of `app.py` `ENABLETWEETS` must be set to True.


```sh
ENABLETWEETS = True
```

### Fetching asset prices live

In line 38 of `app.py`, `PROD` must be set to `True`.

```sh
PROD = True
```

### Deploying

#### Google Cloud AppEngine

1. From the project root, run 

```sh
gcloud init
gcloud app deploy
```

#### Docker

1. From the project root, run

```sh
docker compose build
docker compose up -d
```

2. To start or stop the service

```sh
docker compose start
docker compose stop
```

3. To take down the service

```sh
docker compose down
```

## Screenshots

![Dashboard Screenshot](https://raw.githubusercontent.com/fpontejos/bc5/main/doc_images/01.png)

![Dashboard Screenshot](https://raw.githubusercontent.com/fpontejos/bc5/main/doc_images/02.png)

![Dashboard Screenshot](https://raw.githubusercontent.com/fpontejos/bc5/main/doc_images/03.png)

![Dashboard Screenshot](https://raw.githubusercontent.com/fpontejos/bc5/main/doc_images/04.png)

![Dashboard Screenshot](https://raw.githubusercontent.com/fpontejos/bc5/main/doc_images/05.png)

![Dashboard Screenshot](https://raw.githubusercontent.com/fpontejos/bc5/main/doc_images/06.png)

![Dashboard Screenshot](https://raw.githubusercontent.com/fpontejos/bc5/main/doc_images/07.png)

![Dashboard Screenshot](https://raw.githubusercontent.com/fpontejos/bc5/main/doc_images/08.png)

![Dashboard Screenshot](https://raw.githubusercontent.com/fpontejos/bc5/main/doc_images/09.png)

![Dashboard Screenshot](https://raw.githubusercontent.com/fpontejos/bc5/main/doc_images/10.png)

![Dashboard Screenshot](https://raw.githubusercontent.com/fpontejos/bc5/main/doc_images/11.png)

![Dashboard Screenshot](https://raw.githubusercontent.com/fpontejos/bc5/main/doc_images/12.png)

![Dashboard Screenshot](https://raw.githubusercontent.com/fpontejos/bc5/main/doc_images/13.png)

![Dashboard Screenshot](https://raw.githubusercontent.com/fpontejos/bc5/main/doc_images/14.png)
