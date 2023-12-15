#!/usr/bin/env python

import os
import json
import redis
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from linkextractor import columnas
import numpy as np
from scipy.spatial.distance import cityblock

app = FastAPI()
origins = ["*"]  # Puedes ajustar esto según tus necesidades de CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

redis_conn = redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379"))

@app.get("/")
def index():
    return "Usage: http://<hostname>[:<prt>]/api/<url>"

#----------------------------------------------------------------
total = {}
valoresfinal = {}
peliculasp = {}
df = pd.DataFrame()
csv_path = '/shared_data/movie.csv'

@app.post('/api/csv')
async def recibir_csv(obj: dict):
    global df
    df = pd.DataFrame(obj)
    csv_path = '/shared_data/movie.csv'
    df.to_csv(csv_path, index=False)
    redis_conn.set('csv', json.dumps(obj))
    return {"csv cargado correctamente a redis"}

@app.post('/api/valor')
async def recibir_datos(data: dict):
    global valoresfinal, peliculasp
    col1 = data.get('col1')
    col2 = data.get('col2')
    col3 = data.get('col3')
    numero = data.get('numero')
    numerox = int(numero)

    try:
        csv_path = '/shared_data/movie.csv'
        af = pd.read_csv(csv_path)

        peli = af
        peli[col3] = pd.to_numeric(peli[col3], errors='coerce')
        peli[col1] = pd.to_numeric(peli[col1], errors='coerce')

        consolidated_dfmi = columnas(peli, col1, col2, col3)
        consolidated_dfmi = pd.concat([consolidated_dfmi.query(f'userId == {numerox}'), consolidated_dfmi.head(1000)])
        consolidated_dfmi = consolidated_dfmi.loc[~consolidated_dfmi.index.duplicated(keep='first')]
        consolidated_dfmi = consolidated_dfmi.fillna(0)

        def computeNearestNeighbor(dataframe, target_user, distance_metric=cityblock):
            distances = np.zeros(len(dataframe))
            target_row = dataframe.loc[target_user]
            for i, (index, row) in enumerate(dataframe.iterrows()):
                if index == target_user:
                    continue
                non_zero_values = (target_row != 0) & (row != 0)
                distance = distance_metric(target_row[non_zero_values].fillna(0), row[non_zero_values].fillna(0))
                distances[i] = distance

            sorted_indices = np.argsort(distances)
            sorted_distances = distances[sorted_indices]
            return list(zip(dataframe.index[sorted_indices], sorted_distances))

        target_user_id = numerox
        neighborsmi = computeNearestNeighbor(consolidated_dfmi, target_user_id)
        diccionario_resultante = dict(neighborsmi)
        valoresfinal = diccionario_resultante

        cd2 = pd.DataFrame(neighborsmi)
        cd2.columns = ['Id_user', 'Distancias']

        primeros = cd2['Id_user'].unique().tolist()[:10]
        resul = peli.query('userId in @primeros')
        newx = resul.query('rating == 5.0')['movieId'].drop_duplicates()
        dictionary_final = dict(zip(newx.index, newx.values))
        peliculasp = dictionary_final

        redis_conn.set('valoresfinal', json.dumps(valoresfinal))
        redis_conn.set('peliculas', json.dumps(peliculasp))

        return valoresfinal
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get('/api/valor')
async def get_users():
    cached_data = redis_conn.get('valoresfinal')
    if cached_data:
        return json.loads(cached_data)
    else:
        raise HTTPException(status_code=404, detail="No hay valores finales almacenados en Redis")

@app.get('/api/peliculas')
async def get_peliculas():
    peliculas_cached = redis_conn.get('peliculas')
    if peliculas_cached:
        return json.loads(peliculas_cached)
    else:
        raise HTTPException(status_code=404, detail="No hay valores finales almacenados en Redis")

@app.get('/api/csv')
async def get_csv():
    csv_cached = redis_conn.get('csv')
    if csv_cached:
        return json.loads(csv_cached)
    else:
        raise HTTPException(status_code=404, detail="No hay valores finales almacenados en Redis")
#----------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=80)
