
import numpy as np
import pandas as pd
import spacy
import matplotlib.pyplot as plt
import seaborn as sns
import sentencepiece
import tensorflow as tf
import tensorflow_hub as hub


def nlp_weighting(input_list):
    print('Start')
    nlp = spacy.load('en')
    newtext = []

    for doc in input_list:
        nlpdoc=nlp(doc)
        tempDoc=''
        for token in nlpdoc:
            if token.is_stop == False:
                tempDoc = tempDoc + ' ' + str(token.lemma_)
                if token.pos_ == 'NOUN':
                    #We triple the strength of Nouns
                    tempDoc = tempDoc + ' ' + str(token.lemma_)
                    tempDoc = tempDoc + ' ' + str(token.lemma_)
                elif token.pos_ == 'PROPN':
                    tempDoc = tempDoc + ' ' + str(token.lemma_)
                    tempDoc = tempDoc + ' ' + str(token.lemma_)
                elif  token.pos_ == 'ADJ':
                    #We double the strength fo Adjectives
                    tempDoc = tempDoc + ' ' + str(token.lemma_)
                elif token.pos_ == 'VERB':
                    #We double the strength of Verbs
                    tempDoc = tempDoc + ' ' + str(token.lemma_)
                    
        #Here we have a hard cutoff at 2100 characters. THis is because there were memory issues with the encoding otherwise
        if len(tempDoc) > 2100:
            tempDoc = tempDoc[0:2100]
        if len(tempDoc) < 110:
            tempDoc =''

        newtext.append(tempDoc)
        
    print('Returned')
        
    return(newtext)


def embed(text):
    print('Start')
    print('Starting embeddings...')
    #embed_US = hub.Module("universal_sentence")
    embed_US = hub.Module("https://tfhub.dev/google/universal-sentence-encoder/2")
    embeddings = embed_US(text)
    print('Extracting embeddings...')
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer())
        embd = sess.run(embeddings)
    dim_vector = ['Dim_{}'.format(i) for i in range(embd.shape[1])]
    df_return = pd.DataFrame(embd, columns = dim_vector)
    return df_return