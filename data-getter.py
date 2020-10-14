import praw as pw 
import pandas as pd 



reddit = pw.Reddit(user_agent='aita-data', site_name='main1')


def get_data(red, sub, lim, sort): 

    data = { 
            'title': [],
            'body': [], 
            'judge': [] 
            } 

    subreddit = red.subreddit(sub)
    
    if sort == 'top': 
        subreddit = subreddit.top(limit=lim)
    elif sort == 'hot': 
        subreddit = subreddit.hot(limit=lim) 
    else: 
        subreddit = subreddit.new(limit=lim)


    for post in subreddit: 
        

        flair = post.link_flair_text 

        if flair == 'Not the A-hole': 
            data['judge'].append('0') 
        elif flair == 'Asshole': 
            data['judge'].append('1') 
        else: 
            continue

        data['title'].append(post.title) 
        data['body'].append(post.selftext) 

    return data 

def to_data_frame(data): 

    index = [f'post { i }' for i in range(len(data['judge']))] 

    return pd.DataFrame(data, index=index) 

if __name__ == '__main__': 
    sub = 'AmItheAsshole' 
    sort = 'top'  
    lim = 1000
    data = get_data(reddit, sub, lim, sort) 
    df = to_data_frame(data) 

    with open('data/data.csv', 'wt') as out: 
        df.to_csv(out) 

    yta = df[df['judge'].astype(int) == 1]
    nta = df[df['judge'].astype(int) == 0].sample(len(yta), random_state=42)
    df_trunc = pd.concat([nta, yta])

    with open('data/trunc.csv', 'wt') as out_trunc: 
        df_trunc.to_csv(out_trunc) 







