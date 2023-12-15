#!/usr/bin/env python
from fastapi import FastAPI, HTTPException
from typing import Dict
import pandas as pd

app = FastAPI()



users = {"Angelica": {"Blues Traveler": 3.5, "Broken Bells": 2.0,
 "Norah Jones": 4.5, "Phoenix": 5.0,
 "Slightly Stoopid": 1.5,
 "The Strokes": 2.5, "Vampire Weekend": 2.0},
 "Bill": {"Blues Traveler": 2.0, "Broken Bells": 3.5,
 "Deadmau5": 4.0, "Phoenix": 2.0,
 "Slightly Stoopid": 3.5, "Vampire Weekend": 3.0},
 "Chan": {"Blues Traveler": 5.0, "Broken Bells": 1.0,
 "Deadmau5": 1.0, "Norah Jones": 3.0,
 "Phoenix": 5, "Slightly Stoopid": 1.0},
 "Dan": {"Blues Traveler": 3.0, "Broken Bells": 4.0,
 "Deadmau5": 4.5, "Phoenix": 3.0,
 "Slightly Stoopid": 4.5, "The Strokes": 4.0,
 "Vampire Weekend": 2.0},
 "Hailey": {"Broken Bells": 4.0, "Deadmau5": 1.0,
 "Norah Jones": 4.0, "The Strokes": 4.0,
 "Vampire Weekend": 1.0},
 "Jordyn": {"Broken Bells": 4.5, "Deadmau5": 4.0, "Norah Jones": 5.0,
 "Phoenix": 5.0, "Slightly Stoopid": 4.5,
 "The Strokes": 4.0, "Vampire Weekend": 4.0},
 "Sam": {"Blues Traveler": 5.0, "Broken Bells": 2.0,
 "Norah Jones": 3.0, "Phoenix": 5.0,
 "Slightly Stoopid": 4.0, "The Strokes": 5.0},
 "Veronica": {"Blues Traveler": 3.0, "Norah Jones": 5.0,
 "Phoenix": 4.0, "Slightly Stoopid": 2.5,
 "The Strokes": 3.0}}



valores = pd.DataFrame(users)
valores = valores.fillna(0)
users_dict = valores.to_dict(orient='index')




def columnas(df , a1 ,a2 ,x):
    # Group by 'userId' and 'movieId' and calculate the mean of 'rating'
    consolidated_df1 = df.groupby([a1, a2])[x].mean().unstack()
    return consolidated_df1



'''peliculas = pd.read_csv("peli.csv" , sep=",")
newpe = peliculas.fillna(0)

def coseno(name):
  dot_product = 0
  magnitude_a = 0
  magnitude_b = 0
  suma = 0
  for index , row in newpe.iterrows():
      suma += 1
      if (row.vanessa ==  0 or row[name] == 0 ):
        dot_product = dot_product
 
      else :
        dot_product += row.vanessa * row[name]
        magnitude_a += row.vanessa ** 2
        magnitude_b += row[name]  ** 2
  magnitude_a = magnitude_a ** 0.5 
  magnitude_b = magnitude_b ** 0.5
  #print(suma)
  cosine_similarity = dot_product / (magnitude_a * magnitude_b)
  return cosine_similarity

nombres = [columna for columna in newpe.columns if columna not in ["Unnamed: 0", "vanessa"]]
valores = {}
for x in nombres:
    r = coseno(x)  # Reemplaza 'coseno(x)' con la función que desees utilizar
    valores[x] = r'''
