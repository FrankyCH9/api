#!/usr/bin/env python


from flask import Flask, request, jsonify
import pandas as pd
from flask_cors import CORS


app = Flask(__name__)
CORS(app)



total = {}
valoresfinal = {}




@app.route('/api/valor', methods=['POST'])
def recibir_datos():
    if request.method == 'POST':
        data = request.get_json()  # Obtiene los datos en formato JSON desde la solicitud
        nombre = data.get('obj')  # Suponiendo que esperas un campo 'nombre' en los datos

        # Haz algo con el nombre, por ejemplo, imprímelo
        #print(f"Nombre recibido: {nombre}")
        #total.update(nombre)
        total.update(nombre)
        valores = pd.DataFrame(total)
        valores = valores.fillna(0)

        def coseno(name):
            dot_product = 0
            magnitude_a = 0
            magnitude_b = 0
            suma = 0
            for index , row in valores.iterrows():
                suma += 1
                if (row.Angelica ==  0 or row[name] == 0 ):
                    dot_product = dot_product
            
                else :
                    dot_product += row.Angelica  * row[name]
                    magnitude_a += row.Angelica  ** 2
                    magnitude_b += row[name]  ** 2
            magnitude_a = magnitude_a ** 0.5 
            magnitude_b = magnitude_b ** 0.5
            #print(suma)
            cosine_similarity = dot_product / (magnitude_a * magnitude_b)
            return cosine_similarity

        nombres = [columna for columna in valores.columns if columna not in ["Unnamed: 0", "Angelica"]]
        
        for x in nombres:
            r = coseno(x)  # Reemplaza 'coseno(x)' con la función que desees utilizar
            valoresfinal[x] = r


        return jsonify(nombre)
    else:
        return jsonify({"mensaje": "Esta ruta solo acepta solicitudes POST"})




@app.route('/api/valor', methods=['GET'])
def get_valores():
    #return jsonify(total)
    users_data = valoresfinal
    return jsonify(users_data)




app.run(host="0.0.0.0")
