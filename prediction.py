from twitter import twitter
from tweepy.error import TweepError
import pickle,json
import re
import keras
from nltk.stem.snowball import SnowballStemmer

def main():
    while(True):
        print("\n1> Analyse a person profile")
        print("2> Analyse tweet replies")
        print("3> Analyse tweet replies(only verified)")
        print("4> Analyse my tweet")
        print("5> Analyse search")
        print("6> Analyse hashtag")
        print("7> Report spam")
        print("8> Exit")
        choice = int(input("Enter you choice: "))
        if(choice == 1):
            analyse_profile()
        elif(choice == 2):
            analyze_replies()
        elif(choice == 3):
            analyze_replies(verified=True)
        elif(choice == 4):
            analyse_tweet()
        elif(choice == 5):
            analyze_search("Search")
        elif(choice == 6):
            analyze_search("Hashtag")
        elif(choice == 7):
            report_spam()
        elif(choice == 8):
            break

def analyse_tweets(tweets):
    count = len(tweets)
    if(count==0):
        print("No tweets found!")
        return
    elif(count == 1):
        print("Analysing %d tweet."%count)
    else:
        print("Analysing %d tweets."%count)
    tweets_pre = preprocess(tweets)
    res = predict(tweets_pre,count)
    for key in res:
        print("[{}] POS: {:.2f}%, NEG: {:.2f}%".format(key,res[key][1]/count*100,res[key][0]/count*100))

def analyse_profile():
    name = input("Enter username: ")
    print("Downloading tweets of %s"%name)
    tweets = twitter.get_user_tweets(name)
    print("User profile analyses:")
    analyse_tweets(tweets)

def analyze_replies(verified=False):
    tweet_id = input("Enter tweet ID: ")
    print("Downloading replies of tweet ID: %s"%tweet_id)
    tweets = twitter.get_tweet_replies(tweet_id,verified=verified)
    print("Tweet replies analyses:")
    analyse_tweets(tweets)

def analyse_tweet():
    tweets = [input("Enter tweet: ")]
    print("Tweet analyses:")
    analyse_tweets(tweets)

def analyze_search(text):
    search = input("Enter "+text+": ")
    print("Downloading tweets of %s"%search)
    tweets = twitter.search_tweets(search)
    print(text+" analyses:")
    analyse_tweets(tweets)

def report_spam():
    username = input("Enter username: ")
    try:
        msg = twitter.api.report_spam(username)
        print(msg)
    except TweepError as e:
        print("Wrong username!")

def preprocess(tweets):
    vectorizer = pickle.load(open("vectorizer.pkl","rb"))
    tweets = [clean_tweet(tweet) for tweet in tweets]
    tweets = vectorizer.transform(tweets)
    return tweets    

def predict(tweets,count):
    clf = pickle.load(open("models.pkl","rb"))
    res = {}
    #basic predictors
    for key in clf:
        y = clf[key].predict(tweets)
        res[key] = {}
        res[key][1] = sum(y)
        res[key][0] = count - res[key][1]

    tweets = tweets.toarray()
    #Deep neural network
    mlp = keras.models.load_model("mlp.h5")
    y = mlp.predict(tweets).argmax(axis=1)
    res["DNN"] = {}
    res["DNN"][1] = sum(y)
    res["DNN"][0] = count - res["DNN"][1]

    #Convolutional neural network
    cnn = keras.models.load_model("cnn.h5")
    y = cnn.predict(tweets.reshape(tweets.shape[0],tweets.shape[1],1)).argmax(axis=1)
    res["CNN"] = {}
    res["CNN"][1] = sum(y)
    res["CNN"][0] = count - res["CNN"][1]

    #Recurrent neural network
    rnn = keras.models.load_model("rnn.h5")
    y = rnn.predict(tweets.reshape(tweets.shape[0],1,tweets.shape[1])).argmax(axis=1)
    res["RNN"] = {}
    res["RNN"][1] = sum(y)
    res["RNN"][0] = count - res["RNN"][1]
    return res


def clean_tweet(tweet):
    stemmer = SnowballStemmer('english')
    tweet = re.sub("(@[A-Za-z0-9_]+)|([^A-Za-z \t]) |(\w+:\/\/\S+)", " ", tweet.lower())
    return ' '.join([stemmer.stem(word) for word in tweet.split()])

if __name__=="__main__":
    main()