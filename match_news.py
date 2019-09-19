
import numpy as np
import pandas as pd
from newspaper import Article
from Support_functions import nlp_weighting
from Support_functions import embed
from Generate_Lyrics_Embeddings import generate_lyric_embeddings
from sklearn.metrics.pairwise import cosine_similarity
import os.path
from os import path


article_url = input('Please enter the URL of your Article: ')


if not path.exists('Lyric_embeddings.csv'):
    generate_embed = input('\nGenerating the embeddings for all lyrics takes a very long time...continue? (Y/N):  ')
    if generate_embed == 'Y':
        generate_lyric_embeddings()



df_songs_political_lyrics_embed = pd.read_csv('Lyric_embeddings.csv')
# #First we want to import the particular article that we want

article = Article(article_url)
article.download()
article.parse()
article_text=article.text

#Apply NLP and embed
article_text = nlp_weighting([article_text])
article_text_embed = embed([article_text[0],'temp']).iloc[0,:]

#Run through all embeddings and look for the most similar one
print('Checking all embeddings...this will take a minute')
print('\n')
max_cos = 0
max_col = ''
for i in range(len(df_songs_political_lyrics_embed)):
    temp_cos = cosine_similarity([article_text_embed],[df_songs_political_lyrics_embed.iloc[i]])
    if temp_cos > max_cos:
        max_cos = temp_cos
        max_col = df_songs_political.iloc[i]


print('The name of the closest Song: '+str(max_col[1]))
print('The artist of the closest Song: '+str(max_col[0]))
print('The lyrics of the closest Song: \n'+str(max_col[3]))

