from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.decomposition import PCA
from nltk.tokenize.casual import casual_tokenize 
from nltk.stem.porter import PorterStemmer 
from nltk.corpus import stopwords 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA 
from sklearn.model_selection import train_test_split as tts 
import pandas as pd 
import numpy as np 



def prepare_data(data): # documents 
    
    docs = [title.lower() + body.lower() for title, body in zip(data['title'], data['body'])]
    
    words = stopwords.words('english') 

    tfidf = TfidfVectorizer(tokenizer=casual_tokenize, stop_words=words)
    tfidf_docs = tfidf.fit_transform(raw_documents=docs).toarray()

    new_index = [] 

    for i, judge in zip(range(len(data.index)), data['judge'].to_numpy()): 
        extension = '' 
        if data['judge'][i]: 
            extension = 'A' 

        new_index.append(f'post { i } { extension }')

    df = pd.DataFrame(tfidf_docs, columns=tfidf.vocabulary_, index=new_index) 

    return df 
    

def pca_topic_vectorize(tfidf, components=50): 

    pca = PCA(n_components=components, random_state=20)
    pca_topic_vectors = pca.fit_transform(tfidf) 
    
    columns = [f'topic{ i }' for i in range (pca.n_components)]

    pca_topic_vectors = pd.DataFrame(pca_topic_vectors, columns=columns, index=tfidf.index) 
    pca_topic_vectors = (pca_topic_vectors.T / np.linalg.norm(pca_topic_vectors, axis=1)).T # normalize vecotrs a / |a| 

    return pca_topic_vectors 

def get_trunc(): 
    with open('data/trunc.csv', 'rb') as fl: 
        df = pd.read_csv(fl) 
    return df 

def get_data(): 
    with open('data/data.csv', 'rb') as fl: 
        df = pd.read_csv(fl) 
    return df 


if __name__ == '__main__': 
    data = get_trunc() 
    tfidf = prepare_data(data) 
    topic_vectors = pca_topic_vectorize(tfidf, components=100) 

    X_train, X_test, Y_train, Y_test = tts(topic_vectors, data.judge, test_size=0.5, random_state=42) 
    lda = LDA(n_components=1)
    lda = lda.fit(X_train, Y_train)
    
    print(round(float(lda.score(X_test, Y_test)), 2))

    


