import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df['rating'] = df['rating'].fillna(6.3)
df['type'] = df['type'].fillna(df['type'].mode().values[0])
df['genre']  = df['genre'].fillna(df['genre'].mode().values[0])



df=df[df['genre'] != 'Hentai']

df['combine']=df['name']+" "+df['genre']+" "+df['type']
df['combine'].str.replace(',',"")


df=df[['anime_id','name','combine']]
df.head()

cv=CountVectorizer()
simmatrix = cv.fit_transform(df["combine"])
simmatrix=simmatrix.toarray()
cosinesim=cosine_similarity(simmatrix)

anisimdf = pd.DataFrame(cosinesim,index=df.name,columns=df.name)


