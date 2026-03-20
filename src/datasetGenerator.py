import pandas as pd
#import numpy as np
import unicodedata
import re
import itertools
import random
from src.errorGenerator import ErrorGenerator
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt
from datasets import Dataset, ClassLabel, Sequence, DatasetDict
from huggingface_hub import HfApi
from pathlib import Path
from .constants import ErrorTag
from src.logger import logging

class DatasetGenerator:
    """
    Clase encargada de la generación del nuevo dataset con errores inyectados a partir de un corpus original.
    Limpia y preprocesa el texto, prepara las features necesarias, divide el dataset en splits y orquesta la inyección de errores utilizando la clase ErrorGenerator.   
    Finalmente, guarda los datasets generados en formato CSV y los sube a Hugging Face, además de generar una gráfica con la distribución de tipos de errores

    Parameters
    ----------
    path_data : str
        Ruta donde se guardarán los archivos CSV generados
    sampling : float
        Proporción del dataset original a utilizar para la generación (entre 0 y 1)
    min_string : int, opcional
        Longitud mínima de las oraciones a incluir (por defecto es 8)
    max_string : int, opcional
        Longitud máxima de las oraciones a incluir (por defecto es 100)
    validation_size : float, opcional
        Proporción del dataset para el conjunto de validación (por defecto es 0.1)
    test_size : float, opcional
        Proporción del dataset para el conjunto de prueba (por defecto es 0.1)
    name_dataset : str, opcional     
        Nombre base para los archivos generados (por defecto es 'dataCorrupted_to_GEC-GED')
    nlp : spacy.lang.es.Spanish
        Modelo de lenguaje de spaCy para procesamiento de texto
    data_source : datasets.Dataset
        Conjunto de datos de Hugging Face a utilizar como fuente
    column_source : str, opcional
        Nombre de la columna en data_source que contiene el texto a procesar (por defecto es

    """

    def __init__(self, path_data, sampling, min_string=8, max_string=100, validation_size=0.1, test_size=0.1, name_dataset='dataCorrupted_to_GEC-GED', nlp=None, data_source=None, column_source="text"):
        self.path_data = path_data
        self.sampling = sampling
        self.min_string = min_string
        self.max_string = max_string
        self.validation_size = validation_size
        self.test_size = test_size
        self.name_dataset = name_dataset
        self.nlp = nlp
        self.data_source = data_source
        self.column_source = column_source

    def __divide_sentences(self, text):
        """
        Funcion auxiliar. Divide el texto en una lista de oraciones usando saltos de línea como delimitadores
        Parameters
        ----------
        text : str
            El texto completo a dividir en oraciones
        Returns
        -------
        list
            Lista de oraciones extraídas del texto
        """
        return  re.split(r'\n\n|\n', text)


    def __filter_sentences(self, sentences):
        """
        Funcion auxiliar. Filtra oraciones excluyendo URLs, rutas de archivos y validando 
        que la cantidad de palabras esté dentro del rango (min_string, max_string)
        
        Parameters
        ----------
        sentences : list    
            Lista de oraciones a filtrar
        Returns
        -------
        list 
            Lista de oraciones que cumplen con los criterios de filtrado
        """
        pattern_exclude = r'(' \
                        r'https?://\S+|' \
                        r'www\.\S+|' \
                        r'[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(?:/\S*)?|' \
                        r'(?:/[A-Za-z0-9._-]+)+|' \
                        r'[A-Za-z]:\\(?:[A-Za-z0-9._-]+\\?)*' \
                        r')'
        return [
            sentence.strip()
            for sentence in sentences
            if not re.search(pattern_exclude, sentence)
            and self.min_string <= len(sentence.split()) <= self.max_string
        ]
    

    def __normalize_sentence(self, text):   
        """
        Funcion auxiliar. Normaliza el texto utilizando el estándar Unicode NFKC
        
        Parameters
        ----------
        text : str  
            El texto a normalizar
        Returns
        -------
        str           
            El texto normalizado según el estándar Unicode NFKC
        """     
        return unicodedata.normalize('NFKC', str(text))
        


    def __preprocess_text(self, text):
        """
        Funcion auxiliar. Realiza el preprocesamiento completo: normalización, 
        división por líneas y filtrado

        Parameters
        ----------
        text : str
            El texto completo a preprocesar 
        Returns
        -------
        list
            Lista de oraciones preprocesadas y filtradas
        """
        # Normalizar el texto
        normalized_text = self.__normalize_sentence(text)
        # Dividir el texto por saltos de línea dobles o simples
        sentences = self.__divide_sentences(normalized_text)

        # Filtrar oraciones:
        # - Que no contengan URLs ni rutas
        # - Que estén dentro del rango de longitud definido
        filtered_sentences = self.__filter_sentences(sentences)
        return filtered_sentences
        
    
    #lee el archivo y a cada linea normmaliza()para luego aplanar el texto y que pueda ser usado
    #n_sentences: cantidad de oraciones para la generacion del conjunto de datos
    #min_string : minima cantidad de caracteres que puede contener la oracion
    #max_string: maxima cantidad de carcteres que puede contener la oracion
    def __prepare_features(self,flat_list):
        """
        Funcion auxiliar. Prepara un DataFrame a partir de una lista de oraciones, inicializando columnas para las anotaciones
        
        Parameters
        ----------
        flat_list : list 
            Lista de oraciones preprocesadas y filtradas para preparar el DataFrame
        Returns
        -------
        pandas.DataFrame
            DataFrame con las oraciones del corpus base y columnas inicializadas para las anotaciones de los errores a inyectar
        """
        #flat_list = list(itertools.chain.from_iterable(map(self.__preprocess_text, texts)))
        # se guardan n cantidad de oraciones(n_sentences) en datafr
        #random.shuffle(flat_list) 
        datafr= pd.DataFrame(flat_list, columns=['sentence'])
        datafr['corrupted']=""
        datafr['tokens']= [list() for _ in range(len(datafr))]
        datafr['error_tags'] = [list() for _ in range(len(datafr))]
        datafr['error_type']= [list() for _ in range(len(datafr))]
        datafr['span'] = [list()for _ in range(len(datafr))]
        datafr['annotation'] = [list()for _ in range(len(datafr))]
        datafr['corrupted_tagged'] = ""
        datafr['aux_corrupted_tagged']=[list()for _ in range(len(datafr))]
        datafr['spaces']=[list()for _ in range(len(datafr))]
        return datafr


    def __generate_splits(self, df, seed=42):
        """
        Funcion auxiliar. Divide un DataFrame en train, validation y test.
        
        Parameters
        ----------  
        df : pandas.DataFrame
            El DataFrame completo a dividir en splits
        seed : int, opcional
            Semilla para la aleatorización al barajar el DataFrame (por defecto es 42)
        Returns
        -------
        dict           
          Diccionario de dataframe: 'train', 'validation' y 'test' 
        """        
        if (self.validation_size + self.test_size) >= 1.0:
            raise ValueError("La suma de validation_size y test_size debe ser menor que 1")

        # Barajar el dataframe completo
        df_shuffled = df.sample(frac=1, random_state=seed).reset_index(drop=True)
        
        # Calcular puntos de corte
        n = len(df_shuffled)
        n_test = int(n * self.test_size)
        n_val = int(n * self.validation_size)
        
        # Dividir en splits
        test_df = df_shuffled.iloc[:n_test]
        val_df = df_shuffled.iloc[n_test : n_test + n_val]
        train_df = df_shuffled.iloc[n_test + n_val:]

        return {
        'train': train_df.reset_index(drop=True),
        'validation': val_df.reset_index(drop=True),
        'test': test_df.reset_index(drop=True),
    }

    def __load_data(self):
        """
        Funcion auxiliar. Carga los datos desde la fuente especificada, conservando la cantidad de datos definida por el parámetro sampling 

        Returns
        -------
        datasets.Dataset
            Conjunto de datos cargado y reducido según el muestreo indicado
        """
        ds = self.data_source
        # Conservar solo la columna definida en self.column_source
        ds = ds.remove_columns([col for col in ds.column_names if col != self.column_source])
        ds = ds.shuffle(seed=123)
        splits=ds.train_test_split(train_size=self.sampling, seed=42)
        return splits['train']
        
    
    """
    carga y prepara el conjunto de datos
    1. carga el dataset de wikipedia en español
    2. normaliza y filtra las oraciones
    3. prepara las features del dataset
    4. genera los splits train, validation y test, considerando el muestreo indicado(sampling)
    Retorna:
    - dict con keys 'train', 'validation', 'test' y valores como datasets preparados
    """
    def __prepare_datafr(self):
        """
        Orquesta la carga, preprocesamiento y preparación del dataset original para la generación de errores sintéticos.
            - Carga el dataset desde la fuente especificada
            - Normaliza y filtra las oraciones del corpus
            - Prepara un DataFrame con las features necesarias para la anotación de errores
            - Divide el DataFrame en splits de entrenamiento, validación y prueba
        Returns
        ----------
        dict
            Diccionario con los splits 'train', 'validation' y 'test' listos para la inyección de errores
        """
        logging.info("Cargando conjunto de datos...")
        ds= self.__load_data()
        logging.info(f"Corpus cargado, tamaño: {len(ds)}")

        logging.info("Preparando nuevo conjunto de datos...")        
        flat_list = list(itertools.chain.from_iterable(map(self.__preprocess_text, ds['text'])))
        # se mezcla el corpus 
        random.shuffle(flat_list)
        new_datafr= self.__prepare_features(flat_list)
           
        logging.info(f"Nuevo conjunto de datos preparado. Tamaño: {len(new_datafr)} filas.")
        splits = self.__generate_splits(new_datafr)     

        return splits

    def generateErrors(self):
        """
        Orquesta la generación de errores sintéticos utilizando la clase ErrorGenerator para cada split del dataset preparado.
            - Carga y prepara el dataset original
            - Crea instancias de ErrorGenerator para cada split
            - Inyecta errores en cada split 

        Returns
        -------
        dict
            Diccionario con los splits 'train', 'validation' y 'test' que contienen los datasets con errores inyectados
        """
        dataset= self.__prepare_datafr()
        errorGenerator_train = ErrorGenerator(dataset['train'], self.nlp, error_rate=3)
        errorGenerator_validation = ErrorGenerator(dataset['validation'], self.nlp, error_rate=3)
        errorGenerator_test = ErrorGenerator(dataset['test'], self.nlp, error_rate=3)

        datafr_errors_train = errorGenerator_train.inyect_data_errors()
        datafr_errors_validation= errorGenerator_validation.inyect_data_errors()
        datafr_errors_test = errorGenerator_test.inyect_data_errors()
        return {
            'train':datafr_errors_train,
            'validation':datafr_errors_validation,
            'test':datafr_errors_test
        }    

           
    def save_data_to_csv(self, datafr,split):
        """
        Guarda un DataFrame en formato CSV en la ruta especificada, eliminando columnas auxiliares antes de guardar
        Parameters
        ----------
        datafr : pandas.DataFrame
            El DataFrame a guardar en formato CSV
        split : str
            El nombre del split (train, validation o test) para nombrar el archivo CSV
        """
        logging.info("Guardando conjunto de datos limpio")
        datafr.drop(['spaces','aux_corrupted_tagged'],axis=1, inplace=True)
        path_name=Path(f"{self.path_data}{self.name_dataset}_{split}.csv")
        datafr.to_csv(path_name, index=False)
        logging.info(f"Guardado en {path_name}")


    def save_data_to_Dataset_HF(self, datafr):
        """
        Convierte los DataFrames de cada split en objetos Dataset de Hugging Face, especificando las características de las columnas y sube el dataset completo a Hugging Face bajo el nombre indicado
        Parameters
        ----------
        datafr : dict
            Diccionario con los splits
        """
        labels_name = [tag.label for tag in ErrorTag]
        error_labels = ClassLabel(names=labels_name)
        
        # Convertir cada dataframe a un Dataset
        hf_dataset = DatasetDict({
            split: Dataset.from_pandas(df) for split, df in datafr.items()
        })
        # Especificar que 'error_tags' es una secuencia de ClassLabels
        hf_dataset = hf_dataset.cast_column('error_tags', Sequence(feature=error_labels))
        print("DatasetHF:",hf_dataset)
  
        api = HfApi()
        api.create_repo(self.name_dataset, repo_type="dataset")
        hf_dataset.push_to_hub(self.name_dataset)
        logging.info(f"Dataset guardado en Hugging Face bajo el nombre: {self.name_dataset}")
        


    def plot_data(self, dataset_dict,name_fig):
        """
        Genera una gráfica de barras que muestra la distribución de tipos de errores en cada split del dataset.
        Parameters
        ----------
        dataset_dict : dict
            Diccionario con los splits del dataset
        nameFig : str
            Nombre del archivo PNG donde se guardará la gráfica generada
        """
        all_data = []

        # Procesar cada split
        for split in dataset_dict:
            df = dataset_dict[split]
            # Aplanar listas de errores y contar
            all_errors = [item for sublist in df['error_type'] for item in sublist]
            count_errors = Counter(all_errors)

            # Crear DataFrame con los errores y el nombre del split
            errors_df = pd.DataFrame.from_dict(count_errors, orient='index', columns=['cantidad']).reset_index()
            errors_df.columns = ['error', 'cantidad']
            errors_df['split'] = split  # Añadir el nombre del split
            print("Errors:",errors_df)
            all_data.append(errors_df)
        # Combinar todos los splits en un solo DataFrame
        final_df = pd.concat(all_data)

        # Graficar
        plt.figure(figsize=(10, 6))
        sns.barplot(x='error', y='cantidad', hue='split', data=final_df)
        plt.xlabel('Tipo de Error')
        plt.ylabel('Cantidad')
        plt.title('Distribución de Tipos de Errores por Split')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{self.path_data}{name_fig}", bbox_inches='tight')


    def run_pipeline(self):
        """
        Ejecuta todo el proceso de generación del dataset con errores sintéticos:
            - Carga y prepara el dataset original
            - Genera el nuevo conjunto de datos corrupto
            - Guarda los datasets generados en formato CSV
            - Convierte y sube el dataset completo a Hugging Face
            - Genera una gráfica con la distribución de tipos de errores
        """
        data = self.generateErrors()
        
        self.save_data_to_csv(data['train'],'train')
        self.save_data_to_csv(data['validation'],'validation')
        self.save_data_to_csv(data['test'],'test')

        self.save_data_to_Dataset_HF(data)
        self.plot_data(data,f"distribucion_De_Errores{self.name_dataset}.png")
        

