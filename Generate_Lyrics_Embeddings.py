
import numpy as np
import pandas as pd
from Support_functions import nlp_weighting
from Support_functions import embed

 
def generate_lyric_embeddings():
    #First we import all of the song Lyrics (this should probably be done in another python script)
    df_songs = pd.DataFrame(pd.read_csv('songdata_10_song.csv'))

    #If you want to subset to only select certain artists, this is the place to do so:

    #political_artists = ['Zac Brown Band',
    #'Ziggy Marley',
    #'The Beatles',
    #'Arrogant Worms',
    #'Billy Joel',
    #'Bob Marley',
    #'Coldplay',
    #'Creedence Clearwater Revival',
    #'Elton John',
    #'Eminem',
    #'Fleetwood Mac',
    #'Garth Brooks',
    #'John Denver',
    #'Kanye West',
    #'Linkin Park',
    #'Lynyrd Skynyrd',
    #'Rage Against The Machine',
    #'Rascal Flatts',
    #'Red Hot Chili Peppers',
    #'System Of A Down',
    #'Tragically Hip',
    #'The White Stripes']
    #df_songs_political = df_songs[df_songs['artist'].isin(political_artists)]

    #If subsetting, comment out the following line:
    df_songs_political = df_songs

    df_songs_political=df_songs_political.dropna(subset=['text'])
    df_songs_political['text'] = df_songs_political['text'].apply(lambda x: x.replace('\n',' '))
    df_songs_politcal_lyrics= list(df_songs_political.iloc[:,3])

    """## Here we begin to implement some of the NLP
    """

    df_songs_politcal_lyrics = nlp_weighting(df_songs_politcal_lyrics)

    """Note below we have the version where we are taking TFIDF Weights. In reality, this would be harder to implement. In practice, we will use a pretrained model that will be able to return a vector to compare similarities. Also, we will want to restrict the size of our data in order to make comparisons feasible.
    """
    df_songs_political_lyrics_embed = embed(df_songs_politcal_lyrics)

    df_songs_political_lyrics_embed.to_csv('Lyric_embeddings.csv')
