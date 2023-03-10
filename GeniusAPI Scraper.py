from __future__ import print_function
import io
import os
import sys
import string
import numpy as np
import pandas as pd
import lyricsgenius
import requests 
import json

from tensorflow import keras
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import LambdaCallback, ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, Activation, LSTM, Bidirectional, Embedding

def gather_lyrics(album_name, file_name):
    #API Key
    genius = lyricsgenius.Genius("AyJ1umCfcgGDDRRflIyf5qcHy2WvVf8X_joJtp7mYK_tiqJrQyZTiooA8UCmWMVR")
    song_dict = {"Song Name":[], "Lyrics": []}

    f = open(file_name)
    data = json.load(f)
    parse = json.dumps(data)
    parsed = json.loads(parse)

    for i in parsed['tracks']:
        song_dict["Song Name"].append(i['song']['title'])
        song_dict["Lyrics"].append(i['song']['lyrics'])

    return song_dict

def split_text(x):

   text = x['Lyrics']
   translator = str.maketrans('', '', string.punctuation)
   sections = text.split('\\n\\n')
   keys = {'Intro': np.nan,'Verse 1': np.nan,'Verse 2':np.nan,'Verse 3':np.nan,'Verse 4':np.nan, 'Chorus':np.nan, 'Refrain':np.nan}
   lyrics = str()
   single_text = []
   res = {}

   for s in sections:
       key = s[s.find('[') + 1:s.find(']')].strip()
       if ':' in key:
           key = key[:key.find(':')]          
       if key in keys:
           single_text += [x.lower().replace('(','').replace(')','').translate(translator) for x in s[s.find(']')+1:].split('\\n') if len(x) > 1]   
       res['single_text'] =  ' \n '.join(single_text)

   return pd.Series(res)



def main():

    songs = ["More Life", "Thank Me Later", "Take Care", "Nothing Was The Same", "Views", "Scorpion", "Certified Lover Boy", "Honestly Nevermind", "Her Loss", "Room For Improvement", "Comeback Season", "So Far Gone", "If Youre Reading This Its Too Late", "What A Time To Be Alive"]

    song_dict = {"Song Name":[], "Lyrics": []}
    df = pd.DataFrame(song_dict)
    temp_dict = {}
    for i in songs:
        filename = "Lyrics_" + i.replace(" ", "") + ".json"
        print(filename)
        temp_dict = gather_lyrics(i, filename)
        df2 = pd.DataFrame(temp_dict)
        print(df2)
        df = df.append(df2, ignore_index = True)
        print(df)
        
    df.to_csv('data.csv', index=False)

    dfn = pd.read_csv("data.csv")
    ser = dfn.iloc[:,1]
    dfn['Lyrics'] = dfn['Lyrics'].str.split('Lyrics').str[1]
    dfn['Lyrics'] = dfn['Lyrics'].str.split('See Drake').str[0]
    dfn['Lyrics'] = dfn['Lyrics'].str.split('Top story').str[0]
    cell = dfn.loc[5]["Lyrics"]
    dfn = dfn[dfn['Lyrics'].notnull()]
    print(dfn)



    dfn = dfn.join(dfn.apply(split_text, axis=1))
    print(dfn)
    dfn.to_csv('data cleaned.csv')
    dft = pd.read_csv("data cleaned.csv")
    print(dft)
    

main()

