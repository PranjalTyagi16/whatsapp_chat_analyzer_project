from urlextract import URLExtract
extract=URLExtract()
from wordcloud import WordCloud
import pandas as pd
from collections import Counter
import emoji
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

def fetch_stats(selected_user,df):

    if selected_user=='OverAll':
        #fetches number of messages
        num_messages= df.shape[0]
        #fetches number of words
        words=[]
        for message in df['message']:
            words.extend(message.split())
        #fetch number of media
        num_media_messages = df[df['message'] == '<Media omitted>\n'].shape[0]
        #fetch number of links shared
        links=[]
        for message in df['message']:
                links.extend(extract.find_urls(message))

        return num_messages,len(words),num_media_messages,len(links)

    else:
        new_df=df[df['user']==selected_user]
        num_messages= new_df.shape[0]
        words = []
        for message in new_df['message']:
            words.extend(message.split())
        #fetch number of media
        num_media_messages=new_df[new_df['message']=='<Media omitted>\n'].shape[0]
        #fetch number of links
        links=[]
        for message in new_df['message']:
            links.extend(extract.find_urls(message))
        return num_messages,len(words),num_media_messages,len(links)

def most_busy_users(df):
    x = df['user'].value_counts().head()
    new_df= round((df['user'].value_counts() / df.shape[0]) * 100, 2).reset_index()
    new_df.columns = ['name', 'percent']  # Set column names explicitly
    return x, new_df



def create_wordcloud(selected_user,df):
    f = open('stop_hinglish.txt', 'r')
    stop_words = f.read()

    if selected_user != 'OverAll':
        df = df[df['user'] == selected_user]

    temp = df[df['user'] != 'group_notification']
    temp = temp[temp['message'] != '<Media omitted>\n']

    def remove_stop_words(message):
        y=[]
        for word in message.lower().split():
            if word not in stop_words:
                y.append(word)
        return " ".join(y)

    wc=WordCloud(width=500,height=500,min_font_size=10,background_color='white')
    temp['message']= temp['message'].apply(remove_stop_words)
    df_wc=wc.generate(temp['message'].str.cat(sep=" "))

    return df_wc

def most_common_words(selected_user,df):
    f=open('stop_hinglish.txt','r')
    stop_words=f.read()

    if selected_user!='OverAll':
        df=df[df['user']==selected_user]

    temp=df[df['user']!='group_notification']
    temp=temp[temp['message']!='<Media omitted>\n']

    words=[]
    for message in temp['message']:
        for word in message.lower().split():
            if word not in stop_words:
                words.append(word)
    most_common_df=pd.DataFrame(Counter(words).most_common(20))
    return most_common_df

def emoji_helper(selected_user,df):
    if selected_user!='OverAll':
        df=df[df['user']==selected_user]

    emojis=[]
    for message in df['message']:
        emojis.extend([c for c in message if c in emoji.UNICODE_EMOJI['en']])

    emoji_df=pd.DataFrame(Counter(emojis).most_common(len(Counter(emojis))))

    return emoji_df
def monthly_timeline(selected_user,df):
    if selected_user!='OverAll':
        df=df[df['user']==selected_user]

    timeline=df.groupby(['year','month_num','month']).count()['message'].reset_index()

    time=[]
    for i in range(timeline.shape[0]):
        time.append(timeline['month'][i]+"-"+str(timeline['year'][i]))
    timeline['time']=time #new column is created name time from the list elements
    return timeline

def daily_timeline(selected_user,df):
    if selected_user!='OverAll':
        df=df[df['user']==selected_user]

    daily_timeline=df.groupby(['only_date']).count()['message'].reset_index()

    return daily_timeline

def week_activity_map(selected_user,df):
    if selected_user != 'OverAll':
        df=df[df['user']==selected_user]

    return df['day_name'].value_counts()

def month_activity_map(selected_user,df):
    if selected_user != 'OverAll':
        df=df[df['user']==selected_user]

    return df['month'].value_counts()

def activity_heatmap(selected_user,df):
    if selected_user != 'OverAll':
        df=df[df['user']==selected_user]

    return df.pivot_table(index='day_name',columns='period',values='message',aggfunc='count').fillna(0)















