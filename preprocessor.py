import re
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def preprocess(data):
    pattern = '\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}\s-\s'

    messages = re.split(pattern, data)[1:]
    dates = re.findall(pattern, data)

    df = pd.DataFrame({'user_message': messages, 'message_date': dates})
    df['message_date'] = pd.to_datetime(df['message_date'], format='%d/%m/%Y, %H:%M - ')
    df.rename(columns={'message_date': 'date'}, inplace=True)

    users = []
    messages = []
    sentiments=[]
    #initialize setiments intensity analyzer
    analyzer=SentimentIntensityAnalyzer()
    for message in df['user_message']:
        entry = re.split('([\w\W]+?):\s', message)
        if (entry[1:]):#user name
            users.append(entry[1])
            messages.append(" ".join(entry[2:]))
            # Perform sentiment analysis for each user individually
            sentiment_score = analyzer.polarity_scores(messages[-1])
            sentiment = 'Positive' if sentiment_score['compound'] >= 0 else 'Negative'
            sentiments.append(sentiment)
        else:
            users.append('group_notification')
            messages.append(entry[0])
            sentiments.append('')


    df['user'] = users
    df['message'] = messages
    df['sentiment']=sentiments
    df.drop(columns=['user_message'], inplace=True)

    df['only_date']=df['date'].dt.date
    df['year'] = df['date'].dt.year
    df['month_num']=df['date'].dt.month
    df['month'] = df['date'].dt.month_name()
    df['day'] = df['date'].dt.day
    df['day_name']=df['date'].dt.day_name()
    df['hours'] = df['date'].dt.hour
    df['minute'] = df['date'].dt.minute



    period=[]
    for hour in df[['day_name','hours']]['hours']:
        if hour==23:
            period.append(str(hour)+"-"+str('00'))
        elif hour==0:
            period.append(str(hour)+"-"+str(hour+1))
        else:
            period.append(str(hour)+"-"+str(hour+1))

    df['period']=period

    return df
