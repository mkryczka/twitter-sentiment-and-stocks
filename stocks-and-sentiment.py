
from dataclasses import dataclass
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
import snscrape.modules.twitter as tw
import datetime
import vaderSentiment.vaderSentiment as vader
import yfinance as yf

# Change following variables
scrape = False
daysBack = 360
query = 'CocaCola'
stockName = 'KO'
# Choose existing filename to save tweets if scraping, otherwise choose csv that already exists
folder = r'ENTER FOLDER DIRECTORY'
filename = r'ENTER FILE NAME.csv'
likeWeight = 0.10

@dataclass
class Tweet:
    content: str
    sentiPos: float
    sentiNeu: float
    sentiNeg: float
    sentiCompound: float
    followers: int
    likes: int
    retweets : int
    replies : int
    date: datetime.datetime

tweets = None
dir = folder + filename

# Scrapes tweets until last full day
until = datetime.date.today()
until -= datetime.timedelta(days=1)
since = until - datetime.timedelta(days=daysBack-1)

if (scrape):
    test = tw.TwitterSearchScraper(query + ' lang:en since:'+ str(since))
    analyzer = vader.SentimentIntensityAnalyzer()

    tweetList = []    
    count = 0
    for t in test.get_items():
        senti =  analyzer.polarity_scores(t.content)
        tweetList.append(Tweet(t.content, senti['pos'], senti['neu'],senti['neg'],senti['compound'],t.user.followersCount,t.likeCount,t.retweetCount, t.replyCount, t.date))
        count+=1
        if count % 500 == 0: 
            print(str(count) + ' tweets scraped, date: ' + str(t.date.date()))
    tweets = pd.DataFrame(tweetList)
    tweets.to_csv(dir)
else:
    tweets = pd.read_csv(dir)

tweets['date'] = pd.to_datetime(tweets['date'])

# Tweet sentiment is weighted 1.0 and each like is weighted likeWeight
tweets['likes'] = (tweets['likes']*likeWeight) + 1
tweets['weightedSenti'] = tweets['sentiCompound'] * (tweets['likes'])

# Weekly average sentiment
groupedT = tweets.assign(period=pd.PeriodIndex(tweets['date'], freq='W-Sun')).groupby('period')
groupedT = groupedT['weightedSenti'].mean() / (groupedT['likes']).mean()
groupedT = groupedT.to_numpy()

stock = yf.Ticker(stockName)
since = tweets['date'].iloc[-1].date()
until = tweets['date'].iloc[0].date()
stockPrices = stock.history(start=since,end=until,name='test')

stockPrices['date'] = stockPrices.index

# Weekly average stock price
groupedS = stockPrices.assign(period=pd.PeriodIndex(stockPrices['date'], freq='W-Sun')).groupby('period')['High'].mean()
groupedS = groupedS.to_numpy()

m, b, r, p, se = stats.linregress(groupedT,groupedS)

ax = plt.gca()
plt.scatter(groupedT,groupedS)
plt.plot(groupedT,m*groupedT+b,'r')
plt.title('Stock price vs Sentiment for Twitter Query "' + query + '" using '+ str(len(tweets)) + ' Tweets from ' + str(since) + ' until ' + str(until) )
plt.ylabel('Weekly Average Stock Price ($)')
plt.xlabel('Weekly Average Twitter Sentiment (-1.0 to 1.0)')
plt.text(0.05,0.95,f'r={r:.5f} \np={p:.5f}',transform=ax.transAxes)
plt.show()
