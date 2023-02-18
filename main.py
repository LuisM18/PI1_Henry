import pandas as pd
import glob
import numpy as np
from dask import  dataframe as dd
from dask import delayed


class proceso_ETL:    
    
    def cargar_ratings():
        listacsv = glob.glob('**/[1-8].csv', recursive=True)
        def read_and_label_csv(filename):
            # reads each csv file to a pandas.DataFrame
            df_csv = pd.read_csv(filename)
            
            return df_csv

        # create a list of functions ready to return a pandas.DataFrame
        dfs = [delayed(read_and_label_csv)(fname) for fname in listacsv]
        # using delayed, assemble the pandas.DataFrames into a dask.DataFrame
        ddf = dd.from_delayed(dfs)
        df = ddf.compute()

        return df

    def cargar_peliculas():
        ruta_carpeta = 'MLOpsReviews'
        lista_csv = glob.glob(ruta_carpeta + "/*.csv")

        ids = 'adhn'

        peliculas = pd.DataFrame({})

        for i in range(len(lista_csv)):
            #Leer cada csv
            data = pd.read_csv(lista_csv[i],parse_dates=['date_added'])
            df = pd.DataFrame(data)

            #Cambiar columna 'show_id' por 'id' con identificador de plataforma 
            df.insert(1,'id',ids[i]+df.show_id)
            
            df.rating.fillna('G',inplace=True)

            mask1 = df.rating.str.contains('min|season|Season')
            df.loc[mask1,'duration'] = df.loc[mask1,'rating']
            df.loc[mask1,'rating'] = 'unrated'
            
            mask2 = df.duration.isna()
            df.loc[mask2,'duration'] = np.where(df.loc[mask2,'type'].eq('movie'),'0 min','0 season')    
        
            df[['duration_int','duration_type']] = df.duration.str.split(expand=True)
            df.duration_int = df.duration_int.astype('int')    
            df.drop(columns=['show_id','duration'],inplace=True)

            df = df.apply(lambda x: x.astype(str).str.lower())
            df.duration_type = df.duration_type.replace("seasons","season")

            drating = {'all':'g','nr':'unrated','tv-nr':'unrated','tv-g':'g','pg-13':'13+','16':'16+', 
                    'ages_16_':'16+', 'ages_18_':'18+', 'all_ages':'g','not_rate':'unrated', 
                    'not rated':'unrated', 'tv-y7-fv':'7+','ur':'unrated','nc-17':'18+','tv-pg':'g',
                    'tv-ma':'g','tv-14':'13+','tv-pg':'pg','tv-y7':'7+'}
            df.rating = df.rating.replace(drating)
            
            peliculas = pd.concat([peliculas,df],axis=0)

        peliculas.date_added = pd.to_datetime(peliculas.date_added,format = "%Y-%m-%d")
        peliculas.release_year = peliculas.release_year.astype('int')
        peliculas.duration_int = peliculas.duration_int.astype('int')

        return peliculas

peliculas = proceso_ETL.cargar_peliculas()
ratings = proceso_ETL.cargar_ratings()

from fastapi import FastAPI

app = FastAPI(title="Sistema de recomendacion de peliculas",
                description= "hecho por Luis Miguel Vargas")

@app.get('/')
def docs():
    return app.get("Nada por ahora")
   

@app.get('/duracion')
def get_max_duration(year: int,platform: str,duration_type: str):
    '''Función para obtener pelicula con máxima duración con filtros de
    año, plataforma y tipo de audiovisual(Pelicula o Serie) '''
    platform = platform[0]
    if platform not in 'adhn':
        return 'El campo platform solo admite los valores amazon,disney,hulu o netflix'
    
    if duration_type == 'min':
        df = peliculas[(peliculas.id.str.startswith(platform)) & (peliculas.type== 'movie') & (peliculas.release_year == year)] 
        
    elif duration_type == 'season':
        df = peliculas[(peliculas.id.str.startswith(platform)) & (peliculas.type== 'tv show') & (peliculas.release_year == year)]
    else: 
        return 'El campo duration_type solo admite los valores "min" o "season"'

    idx = df.duration_int.idxmax()        
    return df.title[idx] 
    

@app.get('/peliculas_por_puntaje')
def get_score_count(platform,scored: float,year: int):
    '''Función para obtener el número de  peliculas con un puntaje mayor a un valor especifico
    en un año determinado '''
    platform = platform[0]
    if platform not in 'adhn':
        return 'El campo platform solo admite los valores amazon,disney,hulu o netflix'
    
    ids = peliculas[(peliculas.id.str.startswith(platform)) & (peliculas.release_year == year)].id.values

    rating = ratings.groupby(by='movieId').rating.mean()
    rating = rating[ids]
            
    
    return rating.where(rating > scored).count()

@app.get('/peliculas_plataforma')
def get_platform_count(platform: str):
    '''Función para obtener el número de peliculas disponibles de una plataforma
    '''
    platform = platform[0]
    if platform not in 'adhn':
        return 'El campo platform solo admite los valores amazon,disney,hulu o netflix'
    
    return peliculas[peliculas.id.str.startswith(platform)].id.count()

@app.get('/actor')
def get_actor(platform,year: int):
    ''' Función para obtener el actor que mas interpretaciones ha realizado en una plataforma y año especifico'''
    platform = platform[0]
    if platform not in 'adhn':
        return 'El campo platform solo admite los valores amazon,disney,hulu o netflix'
    
    actores = peliculas[(peliculas.id.str.startswith(platform)) & (peliculas.release_year == year)].cast
    actores = actores.str.split(", ").explode()
    actores = actores.value_counts().drop(labels='nan')

    return actores.index[0]