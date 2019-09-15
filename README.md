# Matching News to Songs

The purpose of this repo is to demonstrate an implementation of the Universal Sentence Encoder, as welll as some experimental NLP techniques in order to get a 'better fit' from the sentence encoder. 

This is implemented by taking in a news article as an input, and comparing it to a series of songs lyrics. The lyrics have also been embedded. Cosine similarity is then taken to get the song that is most similar to the news article.

In the 'visualize embedding space', I also explore the possibility of using TSNE to visualize the song space, looking for similarities between song lyrics to see if they form some natural patterns in the data. Note that this does not have any statistical meaning, but is an exploratory method.
