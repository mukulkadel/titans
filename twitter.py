import tweepy

class Twitter:
    def __init__(self,consumer_token,consumer_secret,access_token,access_token_secret):
        auth = tweepy.OAuthHandler(consumer_token, consumer_secret)
        auth.set_access_token(access_token, access_token_secret)
        self.api = tweepy.API(auth)

    def get_user_tweets(self,name,count=200):
        tweets = []
        for status in tweepy.Cursor(self.api.user_timeline,name).items(count):
            tweets.append(status.text)
        return tweets
    def get_tweet_replies(self,tweet_id,count=30,verified=False):
        replies = []
        main_tweet = self.api.get_status(tweet_id)
        screen_name = main_tweet.author.screen_name
        
        q = "to:"+screen_name
        if(verified==True):
            q += " filter:verified"

        for tweet in tweepy.Cursor(self.api.search,q=q, since_id=tweet_id).items(2000):
            if(len(replies)==count):
                break
            if(tweet.in_reply_to_status_id_str==tweet_id):
                replies.append(tweet.text)
        return replies
    def search_tweets(self,q,count=200):
        tweets = []
        for status in tweepy.Cursor(self.api.search,q=q).items(count):
            tweets.append(status.text)
        return tweets

consumer_token = "consumer_token"
consumer_secret = "consumer_secret"
access_token = 'access_token'
access_token_secret = 'access_token_secret'

twitter = Twitter(consumer_token,consumer_secret,access_token,access_token_secret)