import pandas as pd
#import numpy as np
import unicodedata
import re
import itertools
import random
from src.generator import ErrorGenerator
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt
from datasets import Dataset, ClassLabel, Sequence, DatasetDict
from huggingface_hub import HfApi
from pathlib import Path
import os
from src.logger import logging

class DatasetGenerator:

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
        Divide el texto en una lista de oraciones usando saltos de línea como delimitadores
        """
        return  re.split(r'\n\n|\n', text)

    def __filter_sentences(self, sentences):
        """
        Filtra oraciones excluyendo URLs, rutas de archivos y validando 
        que la cantidad de palabras esté dentro del rango (min_string, max_string)
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
        """Normaliza el texto utilizando el estándar Unicode NFKC"""     
        return unicodedata.normalize('NFKC', str(text))
        
    def __preprocess_text(self, text):
        """
        Realiza el preprocesamiento completo: normalización, 
        división por líneas y filtrado.
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
        Divide un DataFrame en train, validation y test.
        
        Params:
        - df: pd.DataFrame
        - seed: int para reproducibilidad
        """
        
        if (self.validation_size + self.test_size) >= 1.0:
            raise ValueError("La suma de validation_size y test_size debe ser menor que 1")

        # Barajar el dataframe completo
        df_shuffled = df.sample(frac=1, random_state=seed).reset_index(drop=True)
        
        # Calcular puntos de corte
        n = len(df_shuffled)
        n_test = int(n * self.test_size)
        n_val = int(n * self.validation_size)
        
        # Dividir usando slicing
        test_df = df_shuffled.iloc[:n_test]
        val_df = df_shuffled.iloc[n_test : n_test + n_val]
        train_df = df_shuffled.iloc[n_test + n_val:]

        return {
        'train': train_df.reset_index(drop=True),
        'validation': val_df.reset_index(drop=True),
        'test': test_df.reset_index(drop=True),
    }

    def __load_data(self):
        ds = self.data_source
        # Conservar solo la columna 'text'
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
        logging.info("Cargando conjunto de datos...")
        ds= self.__load_data()
        logging.info(f"len corpus cargado: {len(ds)}")

        logging.info("Preparando nuevo conjunto de datos...")        
        flat_list = list(itertools.chain.from_iterable(map(self.__preprocess_text, ds['text'])))
        # se mezcla el corpus 
        random.shuffle(flat_list)
        new_datafr= self.__prepare_features(flat_list)
           
        logging.info(f"Nuevo conjunto de datos preparado. Tamaño: {len(new_datafr)} filas.")
        splits = self.__generate_splits(new_datafr)     

        return splits

    def generateErrors(self):
        dataset= self.__prepare_datafr()
        errorGenerator_train = ErrorGenerator(dataset['train'], self.nlp, error_rate=3)
        errorGenerator_validation = ErrorGenerator(dataset['validation'], self.nlp, error_rate=3)
        errorGenerator_test = ErrorGenerator(dataset['test'], self.nlp, error_rate=3)

        datafr_errors_train = errorGenerator_train.create_dataErrors()
        datafr_errors_validation= errorGenerator_validation.create_dataErrors()
        datafr_errors_test = errorGenerator_test.create_dataErrors()
        return {
            'train':datafr_errors_train,
            'validation':datafr_errors_validation,
            'test':datafr_errors_test
        }    

           
    def save_data_to_csv(self, datafr,split):
        logging.info("Guardando conjunto de datos limpio")
        datafr.drop(['spaces','aux_corrupted_tagged'],axis=1, inplace=True)
        path_name=Path(f"{self.path_data}{self.name_dataset}_{split}.csv")
        datafr.to_csv(path_name, index=False)
        logging.info(f"Guardado en {path_name}")

    def save_data_to_Dataset_HF(self, datafr,labels_name, dataset_name):
        error_labels = ClassLabel(names=labels_name)
        
        # Convertir cada dataframe a un Dataset
        hf_dataset = DatasetDict({
            split: Dataset.from_pandas(df) for split, df in datafr.items()
        })
        # Especificar que 'error_tags' es una secuencia de ClassLabels
        hf_dataset = hf_dataset.cast_column('error_tags', Sequence(feature=error_labels))
        print("DatasetHF:",hf_dataset)
        #hf_dataset =hf_dataset.remove_columns(['span', 'annotation']) #'__index_level_0__']) #
        
        # Cargar tus credenciales de Hugging Face (esto lo hace automáticamente si ya configuraste el token)
        api = HfApi()
        # Subir el Dataset a Hugging Face (esto asume que ya creaste el dataset en Hugging Face)
        api.create_repo(dataset_name, repo_type="dataset")
        hf_dataset.push_to_hub(dataset_name)
        logging.info(f"Dataset guardado en Hugging Face bajo el nombre: {dataset_name}")
        


    def plot_data(self, dataset_dict,nameFig):
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
        plt.savefig(f"{self.path_data}{nameFig}", bbox_inches='tight')


    def run_pipeline(self):
        labels_name = ['O','G-gen','G-nsing','G-nplur','G-verbForm','G-uArt','G-wo','P-missing', 'S-title', 'S-noAccent','S-mistake']

        data = self.generateErrors()
        
        self.save_data_to_csv(data['train'],'train')
        self.save_data_to_csv(data['validation'],'validation')
        self.save_data_to_csv(data['test'],'test')

        self.save_data_to_Dataset_HF(data, labels_name, self.name_dataset)
        self.plot_data(data,f"distribucion_De_Errores{self.name_dataset}.png")
        

