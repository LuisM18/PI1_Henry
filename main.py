import pandas as pd
import glob
import numpy as np
import re
from dask import  dataframe as dd
from dask import delayed
from surprise import dump


class Proceso_ETL:    
    
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

            peliculas = pd.concat([peliculas,df],axis=0)

        peliculas.date_added = pd.to_datetime(peliculas.date_added,format = "%Y-%m-%d")
        peliculas.release_year = peliculas.release_year.astype('int')
        peliculas.duration_int = peliculas.duration_int.astype('int')
        peliculas.title = peliculas.title.str.strip()

        return peliculas
    

class API:    
  
    def __init__(self,peliculas,ratings,modelo): 
        self.peliculas = peliculas
        self.ratings = ratings
        self.modelo = modelo       

    def get_max_duration(self,year: int,platform: str,duration_type: str):
        '''Función para obtener pelicula con máxima duración con filtros de
        año, plataforma y tipo de audiovisual(Pelicula o Serie) '''
        platform = platform[0]
        if platform not in 'adhn':
            return 'El campo platform solo admite los valores amazon,disney,hulu o netflix'
        
        if duration_type == 'min':
            df = self.peliculas[(self.peliculas.id.str.startswith(platform)) & (self.peliculas.type== 'movie') & (self.peliculas.release_year == year)] 
            
        elif duration_type == 'season':
            df = self.peliculas[(self.peliculas.id.str.startswith(platform)) & (self.peliculas.type== 'tv show') & (self.peliculas.release_year == year)]
        else: 
            return 'El campo duration_type solo admite los valores "min" o "season"'

        idx = df.duration_int.idxmax()        
        return df.title[idx] 
        


    def get_score_count(self,platform,scored: float,year: int):
        '''Función para obtener el número de  self.peliculas con un puntaje mayor a un valor especifico
        en un año determinado '''
        platform = platform[0]
        if platform not in 'adhn':
            return 'El campo platform solo admite los valores amazon,disney,hulu o netflix'
        
        ids = self.peliculas[(self.peliculas.id.str.startswith(platform)) & (self.peliculas.release_year == year)].id.values
        muestra = self.ratings.sample(frac=0.01,random_state=0)
        rating = muestra.groupby(by='movieId').rating.mean()
        rating = rating[ids]
                
        
        return rating.where(rating > scored).count()


    def get_platform_count(self,platform: str):
        '''Función para obtener el número de self.peliculas disponibles de una plataforma
        '''
        platform = platform[0]
        if platform not in 'adhn':
            return 'El campo platform solo admite los valores amazon,disney,hulu o netflix'
        
        return self.peliculas[self.peliculas.id.str.startswith(platform)].id.count()


    def get_actor(self,platform,year: int):
        ''' Función para obtener el actor que mas interpretaciones ha realizado en una plataforma y año especifico'''
        platform = platform[0]
        if platform not in 'adhn':
            return 'El campo platform solo admite los valores amazon,disney,hulu o netflix'
        
        actores = self.peliculas[(self.peliculas.id.str.startswith(platform)) & (self.peliculas.release_year == year)].cast
        actores = actores.str.split(", ").explode()
        actores = actores.value_counts().drop(labels='nan')

        return actores.index[0]


    def get_recomendation(self,userid: int,title: str):
        
        title = title.lower()
        if re.search("^\d$",title[2]) != None:
            movieid = title
            if movieid not in self.peliculas.id.values:
                return 'No existe el título especificado'
        else:
            if title not in self.peliculas.title.values:
                return 'No existe el título especificado'
            movieid = self.peliculas.id.where(self.peliculas.title == title)

        pred = self.modelo.predict(userid,movieid).est    

        return True if pred>3.5 else False

   
peliculas = Proceso_ETL.cargar_peliculas()
ratings = Proceso_ETL.cargar_ratings()

pred,modelo = dump.load('modelo_svd')



api = API(peliculas,ratings,modelo) 

app = FastAPI(title="API de peliculas",
                description= "API con información sobre peliculas y programas de televisión y sistema de recomendación para un usuario y titulo dado")


@app.get('/')
def root():
    return "API de PI1 de HenryLabs. Mas informacion en /docs"

@app.get('/max_duration')
def get_max_duration(year: int,platform: str,duration_type: str):
    return api.get_max_duration(year,platform,duration_type)
    
@app.get('/get_score_count')
def get_score_count(platform,scored: float,year: int):
    return api.get_score_count(platform,scored,year)

@app.get('/get_platform_count')
def get_platform_count(platform: str):
    return api.get_platform_count(platform)

@app.get('/get_actor')
def get_actor(platform,year: int):
    return api.get_actor(platform,year)

@app.get('/get_recommendation')
def get_recomendation(userid: int,title: str):
    return api.get_recomendation(userid,title)