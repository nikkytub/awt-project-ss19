import tweepy
import pandas as pd
import csv
import sys
reload(sys)
sys.setdefaultencoding('utf8')

# Please provide your credentials from twitter here
CONSUMER_KEY = ""
CONSUMER_SECRET = ""
OAUTH_TOKEN = ""
OAUTH_TOKEN_SECRET = ""

auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
auth.set_access_token(OAUTH_TOKEN, OAUTH_TOKEN_SECRET)
api = tweepy.API(auth)

public_tweets = api.home_timeline()
for tweet in public_tweets:
    print(tweet.text)


data = pd.read_csv('data.csv', parse_dates=True)

tweet_id = data['Tweet ID'].to_list()
cb = data[' Bullying_Traces?'].to_list()


for i in tweet_id:
    try:
        tweet = api.get_status(i)
        print(i, tweet.text)
        with open('tweets3.csv', 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter=",")
            writer.writerow([i, tweet.text])
    except tweepy.error.TweepError:
        pass
