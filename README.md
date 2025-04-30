# https-github.com-viji2-twitter-sentiment-analysis
import tweepy
import pandas as pd
from datetime import datetime

# Twitter API credentials (replace with your own)
consumer_key = 'your_consumer_key'
consumer_secret = 'your_consumer_secret'
access_token = 'your_access_token'
access_token_secret = 'your_access_token_secret'

def get_twitter_data(keyword, count=100):
    """
    Fetch tweets containing a specific keyword
    """
    # Authenticatei
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth)

    # Collect tweets
    tweets = []
    try:
        fetched_tweets = tweepy.Cursor(api.search_tweets, q=keyword, tweet_mode='extended').items(count)

        for tweet in fetched_tweets:
            text = tweet.full_text
            cleaned_text = ' '.join([word for word in text.split() if not word.startswith('@') and not word.startswith('http')])
            tweets.append({
                'text': cleaned_text,
                'created_at': tweet.created_at,
                'likes': tweet.favorite_count,
                'retweets': tweet.retweet_count
            })

        return pd.DataFrame(tweets)
    except tweepy.TweepError as e:
        print("Error: " + str(e))
        return pd.DataFrame()

# Example usage (we'll use mock data later to avoid API limits)
# df = get_twitter_data("happy", 100)
from textblob import TextBlob
!pip install vaderSentiment
from transformers import pipeline

class SentimentAnalyzer:
    def __init__(self):
        self.vader = SentimentIntensityAnalyzer()
        self.emotion_classifier = pipeline("text-classification",
                                         model="bhadresh-savani/distilbert-base-uncased-emotion",
                                         return_all_scores=True)

    def textblob_sentiment(self, text):
        analysis = TextBlob(text)
        return analysis.sentiment.polarity

    def vader_sentiment(self, text):
        return self.vader.polarity_scores(text)['compound']

    def detect_emotion(self, text):
        emotions = self.emotion_classifier(text)[0]
        # Return the emotion with highest score
        return max(emotions, key=lambda x: x['score'])['label']

    def analyze(self, text):
        return {
            'textblob': self.textblob_sentiment(text),
            'vader': self.vader_sentiment(text),
            'emotion': self.detect_emotion(text)
        }

# Example usage
# analyzer = SentimentAnalyzer()
# print(analyzer.analyze("I'm so happy today!"))
import random
from datetime import datetime, timedelta

def generate_mock_data(years=5):
    emotions = ['joy', 'anger', 'fear', 'sadness', 'surprise', 'love']
    data = []

    for year in range(datetime.now().year - years, datetime.now().year + 1):
        for month in range(1, 13):
            for _ in range(random.randint(50, 100)):  # 50-100 posts per month
                emotion = random.choice(emotions)
                if emotion == 'joy':
                    text = random.choice([
                        f"I'm so happy this year {year}!",
                        "Life is beautiful :)",
                        "What a wonderful day!"
                    ])
                elif emotion == 'anger':
                    text = random.choice([
                        "I hate this so much!",
                        "This makes me furious!",
                        "Why does this always happen??"
                    ])
                elif emotion == 'fear':
                    text = random.choice([
                        "I'm scared about what might happen...",
                        "This is terrifying!",
                        "I don't feel safe anymore"
                    ])
                elif emotion == 'sadness':
                    text = random.choice([
                        "I feel so lonely...",
                        "Why does everything go wrong?",
                        "I can't stop crying"
                    ])
                elif emotion == 'surprise':
                    text = random.choice([
                        "Wow! Didn't see that coming!",
                        "This is amazing!",
                        "I'm shocked!"
                    ])
                else:  # love
                    text = random.choice([
                        "I love this so much!",
                        "You're the best thing that happened to me",
                        "My heart is full of love"
                    ])

                date = datetime(year, month, random.randint(1, 28))
                data.append({
                    'text': text,
                    'created_at': date,
                    'emotion': emotion,
                    'year': year,
                    'month': month
                })

    return pd.DataFrame(data)

# Generate mock data
df = generate_mock_data()
print(df.head())
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from wordcloud import WordCloud

plt.style.use('ggplot')

def plot_emotion_distribution(df):
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='emotion', order=df['emotion'].value_counts().index)
    plt.title('Emotion Distribution')
    plt.xlabel('Emotion')
    plt.ylabel('Count')
    plt.show()

def plot_yearly_trend(df):
    yearly = df.groupby(['year', 'emotion']).size().unstack()
    yearly.plot(kind='line', figsize=(12, 6), marker='o')
    plt.title('Emotion Trends Over Years')
    plt.xlabel('Year')
    plt.ylabel('Count')
    plt.grid(True)
    plt.show()

def plot_monthly_trend(df, year):
    yearly = df[df['year'] == year].groupby(['month', 'emotion']).size().unstack()
    yearly.plot(kind='line', figsize=(12, 6), marker='o')
    plt.title(f'Emotion Trends in {year}')
    plt.xlabel('Month')
    plt.ylabel('Count')
    plt.grid(True)
    plt.show()

def plot_interactive_trend(df):
    fig = px.line(df.groupby(['year', 'emotion']).size().reset_index(name='count'),
                 x='year', y='count', color='emotion',
                 title='Emotion Trends Over Years',
                 labels={'year': 'Year', 'count': 'Number of Posts'})
    fig.show()

def generate_wordcloud(emotion, df):
    text = ' '.join(df[df['emotion'] == emotion]['text'])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(f'Word Cloud for {emotion}')
    plt.axis('off')
    plt.show()

# Run visualizations
plot_emotion_distribution(df)
plot_yearly_trend(df)
plot_interactive_trend(df)
generate_wordcloud('joy', df)
def analyze_custom_text():
    analyzer = SentimentAnalyzer()
    while True:
        text = input("Enter text to analyze (or 'quit' to exit): ")
        if text.lower() == 'quit':
            break
        result = analyzer.analyze(text)
        print("\nAnalysis Results:")
        print(f"TextBlob Sentiment: {result['textblob']:.2f} (-1 to 1 scale)")
        print(f"VADER Sentiment: {result['vader']:.2f} (-1 to 1 scale)")
        print(f"Dominant Emotion: {result['emotion']}\n")

def main():
    print("Emotion Decoding through Sentiment Analysis")
    print("------------------------------------------")

    # Load or generate data
    print("\nGenerating mock data...")
    df = generate_mock_data()
    print(f"Generated {len(df)} social media posts spanning {df['year'].nunique()} years.")

    while True:
        print("\nMenu:")
        print("1. View emotion distribution")
        print("2. View yearly trends")
        print("3. View monthly trends for specific year")
        print("4. Generate word cloud for specific emotion")
        print("5. Analyze custom text")
        print("6. Exit")

        choice = input("Enter your choice (1-6): ")

        if choice == '1':
            plot_emotion_distribution(df)
        elif choice == '2':
            plot_yearly_trend(df)
            plot_interactive_trend(df)
        elif choice == '3':
            year = int(input("Enter year to analyze: "))
            plot_monthly_trend(df, year)
        elif choice == '4':
            emotion = input("Enter emotion (joy, anger, fear, sadness, surprise, love): ")
            generate_wordcloud(emotion, df)
        elif choice == '5':
            analyze_custom_text()
        elif choice == '6':
            print("Exiting program...")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
