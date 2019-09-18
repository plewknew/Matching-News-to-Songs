# Matching News to Songs

The purpose of this repo is to demonstrate two functionalities. Firstly, I want to demonstrate an implementation of the Universal Sentence Encoder, as welll as some experimental NLP techniques in order to get a 'better fit' from the sentence encoder. This is done within the Lyrics_NLP notebook.

This is implemented by taking in a news article as an input, and comparing it to a series of songs lyrics. The lyrics and news article both become embedded. Cosine similarity is then taken to get the song that is most similar to the news article. The NLP techniques that I discussed earlier are twofold:
- First, I got rid of all stop words and lemmatized the words in order to get some basis of word for the model
- Second, I wanted to change the 'weighting' of each of the paragraphs. I thought that potentially Nouns / Proper Nouns were the most important aspect when comparing two paragraphs together, as often these nouns are what the song ends up  being about. In order to weight these higher, I repeat them twice in the text. I also repeat the adjectives and verbs once in order to get a more ideal weighting system


The second thing that I did for this project was to visualize a series of songs from a variety of artists. 

As of right now I have:
- Eminem
- Beyonce
- Bob Marley

Although I want to get lyrics from more politically driven artists if I can find some online. I want to take these lyrics and visualize them at a song level. This means embedding the NLP preprocessed lyrics, and then using TSNE to do feature reduction to two dimensions. The goal of this is to look for similarities between song lyrics to see if they form some natural patterns in the data. It is interesting at a song level, as you can see songs similar to each other lyrically. Note that this does not have any statistical meaning, but is an exploratory method.

Also note that I implemented a sort of 'manual grid search' to visually inspect the TSNE results. Because TSNE does not have a true objective function, I subjectively determined the best hyperparameters to use based on visual inspection.
