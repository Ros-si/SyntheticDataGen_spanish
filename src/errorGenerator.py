import logging
import unidecode
import spacy
from spacy.matcher import Matcher
from spacy.tokens import Doc
import random
import re
from . import constants as c
from src.rules import RulesHandler 

class ErrorGenerator:
    """
    Procesa lotes de texto para inyectar errores controlados, gestionando la actualización de etiquetas, anotaciones y la reconstrucción de oraciones corruptas en un dataFrame.

    Parameters
    ----------
    datafr : pandas.DataFrame
        Dataset que contiene el texto base para la inyección de errores
    nlp : spacy.lang.Language
        Modelo de procesamiento de lenguaje natural de spaCy
    error_rate : int = 3
        Número máximo de errores a inyectar por cada tipo de error en una oración
    rules: RulesHandler
        Manejador de reglas para la inyeccion de errores
    """

    def __init__(self, datafr, nlp, error_rate=3):
        self.datafr = datafr
        self.error_rate = error_rate 
        self.nlp = nlp
        self.rules = RulesHandler(nlp=self.nlp)  # Se inyectará el modelo nlp desde el test
    

    def __init_columns(self, idx, doc):
        """
        Inicializa las columnas para una fila específica del DataFrame, si esta aun no tiene errores inyectados

        Parameters
        ----------
        idx : int
            Índice de la fila en el DataFrame
        doc : spacy.tokens.Doc
            Texto procesado por spaCy del cual se extraen los datos iniciales
        """
        if self.datafr['tokens'][idx]:
            return
        else:
            for token in doc: # aun no hay errores generados - se inicia las columnas 
                self.datafr['tokens'][idx].append(token.text)                
                self.datafr['error_tags'][idx].append(c.ErrorTag.NO_ERROR.id_num)
                self.datafr['aux_corrupted_tagged'][idx].append(token.text)
                self.datafr['spaces'][idx].append(bool(token.whitespace_))


    def __get_k_tokens_candidates(self, tokens_candidates):
            """
            Selecciona una muestra aleatoria de tokens candidatos basándose en el error_rate

            Parameters
            ----------
            tokens_candidates : list
                Lista de tokens candidatos para ser corrompidos.

            Returns
            -------
            list
                Subconjunto de tokens seleccionados aleatoriamente.
            """
            k = min(self.error_rate, len(tokens_candidates))
            return random.sample(tokens_candidates, k=k)  
        

    #Agrega el tipo de error, indices de la oracion donde se encuentra el error, y la modificacion realizada(error añadido)
    #agrega errores al conjunto de datos
    def __set_annotation_errors(self, idx_data, span_ini, span_end, t_correct, t_error, tag):
        """
        Registra los metadatos del error inyectado en las columnas correspondientes

        Actualiza el tipo de error, los índices de los caracteres (span) y la 
        anotación (par error/corrección)

        Parameters
        ----------
        idx_data : int
            Índice de la fila
        span_ini : int
            Índice inicial del token/segmento afectado
        span_end : int
            Índice final del token/segmento afectado
        t_correct : str
            Texto original correcto
        t_error : str
            Texto modificado con el error
        tag : str
            Etiqueta del tipo de error 
        """
        self.datafr['error_type'][idx_data].append(tag)
        self.datafr['span'][idx_data].append((span_ini, span_end))
        self.datafr['annotation'][idx_data].append((t_error, t_correct))


    def __update_version_corrupted(self, idx_data):
        """
        Reconstruye y actualiza las columnas 'corrupted' y 'corrupted_tagged'del texto con errores

        Parameters
        ----------
        idx_data : int
            Índice de la fila a procesar
        """
        try:
            self.datafr.loc[idx_data, 'corrupted'] = Doc(self.nlp.vocab, words=self.datafr['tokens'][idx_data], spaces=self.datafr['spaces'][idx_data]).text.strip()
            self.datafr.loc[idx_data, 'corrupted_tagged']= Doc(self.nlp.vocab, words= self.datafr['aux_corrupted_tagged'][idx_data], spaces=self.datafr['spaces'][idx_data]).text.strip()
        except Exception as e:
            print(f"Error al procesar el índice {idx_data} para corrupted_tagged: {e}")
            


    def __get_new_corrupted_tagged(self, idx_data, span_ini, span_text):
        """
        Actualiza un token que ya contiene una etiqueta de error previa

        Parameters
        ----------
        idx_data : int
            Índice de la fila
        span_ini : int
            Posición del token
        span_text : list of str
            Nuevo texto a insertar dentro de la etiqueta existente

        Returns
        -------
        str
            Cadena formateada con la etiqueta y el texto actualizado
        """
        corrupted_tagged = self.datafr['aux_corrupted_tagged'][idx_data][span_ini]
        old = re.search(r"\s([^>]+)>", corrupted_tagged).group(1) # Extrae el texto original del formato "<TAG texto>"
        return corrupted_tagged.replace(old, " ".join(span_text))


    def __update_aux_corrupted_tagged(self, idx_data, span_ini, span_end, span_text, tag):
        """
        Función auxiliar que gestiona la inserción de etiquetas de error en la columna auxiliar 'aux_corrupted_tagged. Aplica lógica especial para errores de orden de palabras y omisión (artículos/puntuación)

        Parameters
        ----------
        idx_data : int
            Índice de la fila
        span_ini : int
            Inicio del segmento afectado
        span_end : int
            Fin del segmento afectado
        span_text : list of str
            Texto o lista de textos corruptos
        tag : str
            Etiqueta del error
        """
        if tag == c.ErrorTag.WORD_ORDER.label:
            self.datafr['aux_corrupted_tagged'][idx_data][span_ini:span_end]= list(self.datafr['aux_corrupted_tagged'][idx_data][span_ini:span_end])[::-1]
        else:
            if self.datafr['error_tags'][idx_data][span_ini] != c.ErrorTag.NO_ERROR.id_num: # ya  tiene errores
                span_text =self.__get_new_corrupted_tagged(idx_data, span_ini, span_text)                        
                self.datafr['aux_corrupted_tagged'][idx_data][span_ini]= span_text  
            else:
                self.datafr['aux_corrupted_tagged'][idx_data][span_ini:span_end]= span_text        
        if tag in [c.ErrorTag.PUNCT_MISSING.label, c.ErrorTag.ART_MISSING.label]: 
            self.datafr['aux_corrupted_tagged'][idx_data][span_ini]= f" <{tag} {self.datafr['aux_corrupted_tagged'][idx_data][span_ini]}"
            self.datafr['aux_corrupted_tagged'][idx_data][span_end-1]=f"{self.datafr['aux_corrupted_tagged'][idx_data][span_end-1]}> "
        else:
            self.datafr['aux_corrupted_tagged'][idx_data][span_ini]= f"<{tag} {self.datafr['aux_corrupted_tagged'][idx_data][span_ini]}"
            self.datafr['aux_corrupted_tagged'][idx_data][span_end-1]=f"{self.datafr['aux_corrupted_tagged'][idx_data][span_end-1]}>"
    
    
    def __update_error_tags_and_tokens(self, idx_data, span_ini, t_error,id_tag):
        """
        Actualiza el token modificado y agrega su ID de error en el DataFrame

        Parameters
        ----------
        idx_data : int
            Índice de la fila
        span_ini : int
            Posición del token a actualizar
        t_error : str
            Nuevo texto (con error)
        id_tag : int
            ID numérico del error
        """
        self.datafr['error_tags'][idx_data][span_ini] = id_tag
        self.datafr['tokens'][idx_data][span_ini] = t_error 


    def __get_entities(self, doc):
        """
        Identifica los tokens que pertenecen a Entidades Nombradas (NER)

        Parameters
        ----------
        doc : spacy.tokens.Doc
            Texto procesado por spaCy

        Returns
        -------
        set
            Conjunto de entidades encontradas en el texto
        """
        return set(tok.text for ent in doc.ents for tok in ent)
    
    #g/genre con articulos, pronombres  y las preposicion del ,de la 
    # se  verifica que el token sea articulo o pronombre, se modifica el token por su contrapuesto en las listas:
    # rt_pron_fem (de genero femenino) y rt_pron_masc (de genero masculino)
    def apply_errors_ggenre(self, batch_,id_ini):
        """
        Inyecta errores de género gramatical en un lote de oraciones

        Busca determinantes y pronombres candidatos, verifica que no sean 
        entidades nombradas y aplica la regla de intercambio de género.

        Parameters
        ----------
        batch_ : list of str
            Lista de oraciones originales.
        id_ini : int
            Índice inicial en el DataFrame para este lote.
        """
        docs = list(self.nlp.pipe(batch_))
        idx_data = id_ini    
        for doc in docs:
            self.__init_columns(idx_data, doc)
            tokens_ents = self.__get_entities(doc) 
            if self.datafr['corrupted'][idx_data]:  #ya tiene errores generados, se parte del texto con errores para generar nuevos errores
                doc= Doc(self.nlp.vocab, words=self.datafr['tokens'][idx_data], spaces=self.datafr['spaces'][idx_data] )
            # Se busca tokens candidatos
            tokens_candidates =[token for token in doc if token.pos_ in ['DET','PRON'] 
                                    and (token.text in c.ART_PRON_MASC or token.text in c.ART_PRON_FEM)
                                    and token.text not in tokens_ents]
            

            if tokens_candidates:     
                random_tokens = self.__get_k_tokens_candidates(tokens_candidates)  # tokens a tomar en cuenta      
                for token in random_tokens:         
                    t_text = self.rules.generate_error_genre(token)
                    self.__set_annotation_errors(idx_data, token.i , token.i+1, token.text, t_text, c.ErrorTag.GENDER.label)      
                    self.__update_aux_corrupted_tagged(idx_data, token.i, token.i+1, [t_text], c.ErrorTag.GENDER.label)  
                    self.__update_error_tags_and_tokens(idx_data, token.i, t_text, c.ErrorTag.GENDER.id_num)
                self.__update_version_corrupted(idx_data)
            idx_data +=1
        
    # se verifica que el token sea  un sustantivo singular, y si termina en una vocal, se o modifica, agregando una 's' al final de la palabra
    def apply_errors_gnumPlur(self, batch_, id_ini):
        """
        Inyecta errores de número plural en sustantivos y adjetivos singulares

        Busca tokens con morfología singular, valida que no sean entidades y aplica reglas de pluralización 

        Parameters
        ----------
        batch_ : list of str
            Lote de oraciones a procesar
        id_ini : int
            Índice del DataFrame donde inicia la inyección de errores para este lote
        """
        docs = list(self.nlp.pipe(batch_))
        idx_data = id_ini        
        for doc in docs:
            self.__init_columns(idx_data, doc)
            tokens_ents = self.__get_entities(doc) 
            
            if self.datafr['corrupted'][idx_data]: # se verifica si ya se tiene errores generados, si es asi ,entonces se toma los tokens generados, alguno ya puede tener algun error, entonce se agregara mas errores sopbre ese
                doc= Doc(self.nlp.vocab, words=self.datafr['tokens'][idx_data], spaces=self.datafr['spaces'][idx_data] )

            tokens_candidates = [token for token in doc if token.pos_ in ['NOUN', 'ADJ'] 
                                      and token.morph.get('Number')==['Sing'] 
                                      and token.text not in tokens_ents ]
            if tokens_candidates:     
                random_tokens = self.__get_k_tokens_candidates(tokens_candidates)  # tokens a tomar en cuenta    
                for token in random_tokens:
                    t_text = self.rules.generate_error_gnumPlur(token)
                    if t_text:
                        self.__set_annotation_errors(idx_data, token.i, token.i+1, token.text, t_text, c.ErrorTag.NUM_PLUR.label )
                        self.__update_aux_corrupted_tagged(idx_data, token.i, token.i+1,[t_text], c.ErrorTag.NUM_PLUR.label)     
                        self.__update_error_tags_and_tokens(idx_data, token.i, t_text, c.ErrorTag.NUM_PLUR.id_num)
                self.__update_version_corrupted(idx_data)
            idx_data += 1


    #g/number singular
    # se modifica los sustantivos y adjetivos plurales a su forma singular 
    def apply_errors_gnumSing(self, batch_, id_ini):
        """
        Inyecta errores de número singular en sustantivos y adjetivos plurales.

        Identifica palabras en plural y las fuerza a su forma base (lema)

        Parameters
        ----------
        batch_ : list of str
            Lote de oraciones
        id_ini : int
            Índice del DataFrame donde inicia la inyección de errores para este lote
        """
        docs = list(self.nlp.pipe(batch_))
        idx_data = id_ini        
        for doc in docs:
            self.__init_columns(idx_data, doc)
            tokens_ents = self.__get_entities(doc) 
            if self.datafr['corrupted'][idx_data]: # se verifica si ya se tiene errores generados, si es asi ,entonces se toma los tokens generados, alguno ya puede tener algun error, entonce se agregara mas errores sopbre ese
                doc= Doc(self.nlp.vocab, words=self.datafr['tokens'][idx_data], spaces=self.datafr['spaces'][idx_data] )

            tokens_candidates = [token for token in doc if token.pos_ in ['NOUN','ADJ'] 
                                      and token.morph.get('Number')==['Plur']
                                      and token.text not in tokens_ents ]
            if tokens_candidates:     
                random_tokens = self.__get_k_tokens_candidates(tokens_candidates)  # tokens a tomar en cuenta           
                for token in random_tokens:
                    t_text = self.rules.generate_error_lemma(token)
                    if t_text:
                        self.__set_annotation_errors(idx_data, token.i, token.i+1, token.text, t_text, c.ErrorTag.NUM_SING.label )
                        self.__update_aux_corrupted_tagged(idx_data, token.i, token.i+1, [t_text], c.ErrorTag.NUM_SING.label)  
                        self.__update_error_tags_and_tokens(idx_data, token.i, t_text, c.ErrorTag.NUM_SING.id_num)
                    else: continue        
                self.__update_version_corrupted(idx_data)
            idx_data += 1



    #g/guart verifica si el token.morph PronType=='Art', es decir un articulo y lo remueve
    def apply_errors_guart(self, batch_, id_ini):
        """
        Induce errores por omisión de artículos en la oración.

        Localiza tokens marcados como artículos (Art) y los remueve, ajustando además los espacios en blanco para mantener la coherencia visual

        Parameters
        ----------
        batch_ : list of str
            Lote de oraciones a procesar
        id_ini : int
            Índice del DataFrame donde inicia la inyección de errores para este lote
        """
        docs = list(self.nlp.pipe(batch_))
        idx_data = id_ini
        for doc in docs:
            self.__init_columns(idx_data, doc)
            tokens_ents = self.__get_entities(doc) 
            idxs_error = [num for span in self.datafr['span'][idx_data] for num in span ]
            tokens_candidates =  [token for token in doc if 'Art' in token.morph.get('PronType')
                                                and token.text not in tokens_ents 
                                                and token.i+1 not in idxs_error 
                                                and token.i-1 not in idxs_error] #tambien aseguramos que no se elimine si un indice anterior o posterior si ya fue modificado
            if tokens_candidates:     
                random_tokens = self.__get_k_tokens_candidates(tokens_candidates)  # tokens a tomar en cuenta               
                for token in random_tokens:
                    t_text = self.rules.generate_error_eliminate()
                    self.__set_annotation_errors(idx_data, token.i, token.i+1, token.text, t_text, c.ErrorTag.ART_MISSING.label )
                    self.__update_aux_corrupted_tagged(idx_data, token.i, token.i+1, [t_text], c.ErrorTag.ART_MISSING.label)     
                    self.__update_error_tags_and_tokens(idx_data, token.i, t_text, c.ErrorTag.ART_MISSING.id_num)             
                    self.datafr['spaces'][idx_data][token.i-1]=False
                    self.datafr['spaces'][idx_data][token.i]=False
                self.__update_version_corrupted(idx_data)
            idx_data += 1



    #g/vtense   modificado el nombre de la funcion a g_verbForm
    #Armin van Buuren  es un nombre , por lo que no deberia identificar van como un verbo, la solucion mas sencilla es ignorar la oracion si contiene entidades PER, que serian nombre
    #se verifica que la oracion no sea una entidad nombrada, 
    #si el token es un verbo o verbo auxiliar, se lo convierte a su forma base, sin conjugaciones ni inflexiones
   
    def apply_errors_gverbForm(self, batch_,id_ini):
        """
        Inyecta errores de forma verbal convirtiendo verbos conjugados a su lema

        Parameters
        ----------
        batch_ : list of str
            Lote de oraciones originales.
        id_ini : int
            Índice de inicio en el DataFrame.
        """
        docs = list(self.nlp.pipe(batch_))
        idx_data = id_ini
        for doc in docs:
            #flag=1
            tokens_ents = self.__get_entities(doc) 
            self.__init_columns(idx_data, doc)
            tokens_candidates = [token for token in doc if (token.pos_=="VERB" or token.pos_=="AUX")
                                                 and token.text not in tokens_ents ]

            if tokens_candidates:       
                random_tokens = self.__get_k_tokens_candidates(tokens_candidates)
                for token in random_tokens:
                    t_lemma = self.rules.generate_error_lemma(token)
                    if t_lemma: 
                        self.__set_annotation_errors(idx_data, token.i, token.i+1, token.text, t_lemma, c.ErrorTag.VERB_FORM.label)
                        self.__update_aux_corrupted_tagged(idx_data,token.i, token.i+1, [t_lemma], c.ErrorTag.VERB_FORM.label)
                        self.__update_error_tags_and_tokens(idx_data, token.i, t_lemma, c.ErrorTag.VERB_FORM.id_num)
                               
                    else: continue
                self.__update_version_corrupted(idx_data)
            idx_data +=1
    

        
    #g/gwo 
    #se utiliza Matcher para encontrar las coincidencias respecto al orden de la palabras en la oracion (patterns_wo)
    #al encontrar los patrones definidos se modifica la oracion invietiendo el orden
    def apply_errors_ggword_order(self, batch_, id_ini):
        """
        Genera errores de orden de palabras mediante patrones POS.

        Utiliza un Matcher para identificar estructuras sintácticas comunes 
        (ej. Sustantivo + Adjetivo) e invierte el orden de los tokens detectados

        Parameters
        ----------
        batch_ : list of str
            Lote de oraciones a procesar
        id_ini : int
            Índice en el DataFrame para este lote
        """
        docs = list(self.nlp.pipe(batch_))
        matcher = Matcher(self.nlp.vocab)
        idx_data = id_ini
        # Agregar patrones al matcher
        for i, pattern in enumerate(c.PATTERNS_WO):
            matcher.add(f'wo_correct{i}', [pattern])
        for doc in docs:               
            self.__init_columns(idx_data, doc)
            tokens_ents = self.__get_entities(doc) 
            all_idxs_matcher=[]
            for _, start, end in matcher(doc):  
                t_correct = self.datafr['tokens'][idx_data][start:end] 
                idxs_matcher = list(range(start,end))
                if any(item in all_idxs_matcher for item in idxs_matcher) or any(token in tokens_ents for token in t_correct): # comprobamos si alguno de los índices del matcher ya fue añadido previamente
                    break  # Si ya fue procesado o es una entidad, saltamos a la siguiente coincidencia
                all_idxs_matcher.extend(idxs_matcher)    
                t_correct_spaces = self.datafr['spaces'][idx_data][start: end]                 
                t_cc_reverse = self.rules.generate_error_ggword_order(t_correct)
                text_correct = Doc(self.nlp.vocab, words=t_correct, spaces=t_correct_spaces)
                t_error = Doc(self.nlp.vocab, words=t_cc_reverse, spaces=t_correct_spaces)         
                self.__set_annotation_errors(idx_data, start, end, text_correct.text, t_error.text, c.ErrorTag.WORD_ORDER.label)  
                self.__update_aux_corrupted_tagged(idx_data, start, end, t_cc_reverse, c.ErrorTag.WORD_ORDER.label)    
                self.datafr['tokens'][idx_data][start:end] = t_cc_reverse
                self.datafr['error_tags'][idx_data][start:end]= [c.ErrorTag.WORD_ORDER.id_num]*(end-start)
                
            self.__update_version_corrupted(idx_data)
            idx_data += 1 

        

    #s/ generate_error_title 
    #Convierte el texto del token a formato título, 
    # donde la primera letra de cada palabra está en mayúscula y las demás en minúscula
    def apply_errors_stitle(self, batch_, id_ini):
        """
        Induce errores de capitalización (formato Título)

        Selecciona tokens que no deberían llevar mayúscula inicial (fuera de entidades)
        y los transforma a Title Case

        Parameters
        ----------
        batch_ : list of str
            Lote de oraciones 
        id_ini : int
            Índice del DataFrame para este lote
        """
        docs = list(self.nlp.pipe(batch_))
        idx_data = id_ini
        for doc in docs:             
            self.__init_columns(idx_data, doc)
            tokens_ents = self.__get_entities(doc) 
            if self.datafr['corrupted'][idx_data]: # se verifica si ya se tiene errores generados, si es asi ,entonces se toma los tokens generados, alguno ya puede tener algun error, entonce se agregara mas errores sopbre ese
                doc= Doc(self.nlp.vocab, words=self.datafr['tokens'][idx_data], spaces=self.datafr['spaces'][idx_data] )

            tokens_candidates = [token for token in doc if token.text !=" " 
                                 and len(token.text) >2 
                                 and not token.is_title and token.text not in tokens_ents ]
            if tokens_candidates:     
                random_tokens = self.__get_k_tokens_candidates(tokens_candidates)  # tokens a tomar en cuenta                
                for token in random_tokens:
                    word_title = self.rules.generate_error_title(token)
                    if word_title:                            
                        self.__set_annotation_errors(idx_data, token.i, token.i+1, token.text, word_title, c.ErrorTag.TITLE.label )
                        self.__update_aux_corrupted_tagged(idx_data, token.i, token.i+1, [word_title], c.ErrorTag.TITLE.label)
                        self.__update_error_tags_and_tokens(idx_data, token.i, word_title, c.ErrorTag.TITLE.id_num)  
                    else: continue                        
                self.__update_version_corrupted(idx_data)
            idx_data += 1
                      

    #s/ generate_error_accent 
    # se usa unicode, para eliminar los acentos y caracteres especiales de una palabra, 
    # primeramente para encontrar palabras con tilde se usa el patron: pattern_accent
    def apply_errors_saccent(self, batch_, id_ini):
        """
        Induce errores de acentuación eliminando tildes de las palabras.

        Utiliza expresiones regulares para identificar tokens que contienen 
        caracteres acentuados y los reemplaza por su versión plana.

        Parameters
        ----------
        batch_ : list of str
            Lote de oraciones 
        id_ini : int
            Índice del DataFrame para este lote
        """
        docs = list(self.nlp.pipe(batch_))
        idx_data = id_ini
        for doc in docs: 
            self.__init_columns(idx_data, doc)
            tokens_ents = self.__get_entities(doc) 
            if self.datafr['corrupted'][idx_data]: # se verifica si ya se tiene errores generados, si es asi ,entonces se toma los tokens generados, alguno ya puede tener algun error, entonce se agregara mas errores sopbre ese
                doc= Doc(self.nlp.vocab, words=self.datafr['tokens'][idx_data], spaces=self.datafr['spaces'][idx_data] )

            tokens_candidates = [token for token in doc if  c.PATTERN_ACCENT.search(token.text) is not None #and token.pos_!="VERB" 
                                            and token.text not in tokens_ents]
            
            if tokens_candidates:     
                random_tokens = self.__get_k_tokens_candidates(tokens_candidates)  # tokens a tomar en cuenta                
                for token in random_tokens:
                    word = self.rules.generate_error_accent(token)
                    self.__set_annotation_errors(idx_data, token.i, token.i+1, token.text, word, c.ErrorTag.ACCENT.label )
                    self.__update_aux_corrupted_tagged(idx_data, token.i, token.i+1, [word], c.ErrorTag.ACCENT.label)
                    self.__update_error_tags_and_tokens(idx_data, token.i, word, c.ErrorTag.ACCENT.id_num)
                self.__update_version_corrupted(idx_data)
            idx_data += 1
    
    #s/ error_substitution
    ##se reemplaza la letra(key) por el correspondiente value in ['VERB','NOUN','ADJ']
    def apply_errors_smistake(self, batch_, id_ini):
        """
        Inyecta errores ortográficos fonéticos (sustitución de letras).

        Busca palabras (sustantivos, verbos, adjetivos) que contengan caracteres 
        susceptibles a confusión fonéticay aplica un reemplazo

        Parameters
        ----------
        batch_ : list of str
            Lote de oraciones a procesar
        id_ini : int
            Índice en el DataFrame para este lote
        """
        docs = list(self.nlp.pipe(batch_))
        idx_data = id_ini
        replacement_keys = list(c.PHONETIC_REPLACEMENTS.keys())
        for doc in docs: 
            self.__init_columns(idx_data, doc)
            tokens_ents = self.__get_entities(doc) 
            if self.datafr['corrupted'][idx_data]: # se verifica si ya se tiene errores generados, si es asi ,entonces se toma los tokens generados, alguno ya puede tener algun error, entonce se agregara mas errores sopbre ese
                doc= Doc(self.nlp.vocab, words=self.datafr['tokens'][idx_data], spaces=self.datafr['spaces'][idx_data] )
            tokens_candidates = [token for token in doc if len(token.text)>3
                                and token.text not in tokens_ents 
                                and token.pos_ in ['VERB','NOUN','ADJ']
                                and any(key in token.text for key in replacement_keys)]
            if tokens_candidates:
                random_candidates = self.__get_k_tokens_candidates(tokens_candidates)  # tokens a tomar en cuenta      
                for token in random_candidates:
                    t_text = self.rules.generate_spelling_mistake(token)
                    self.__set_annotation_errors(idx_data, token.i, token.i+1, token.text, t_text, c.ErrorTag.SPELLING.label )
                    self.__update_aux_corrupted_tagged(idx_data, token.i, token.i+1, [t_text], c.ErrorTag.SPELLING.label)
                    self.__update_error_tags_and_tokens(idx_data, token.i, t_text, c.ErrorTag.SPELLING.id_num)
                self.__update_version_corrupted(idx_data)
            idx_data += 1


    # errors punctuation 
    # se elimina el signo de puntuacion de la oracion
    def apply_errors_punctuation(self, batch_,id_ini):
        """
        Remueve signos de puntuación de las oraciones.

        Identifica tokens de puntuación y los elimina, ajustando los espacios 
        adyacentes para evitar espacios dobles o huérfanos

        Parameters
        ----------
        batch_ : list of str
            Lote de oraciones
        id_ini : int
            Índice inicial en el DataFrame para este lote
        """
        docs = list(self.nlp.pipe(batch_))
        idx_data = id_ini
        for doc in docs:
            self.__init_columns(idx_data, doc)
            tokens_ents = self.__get_entities(doc) 
            idxs_error = [num for span in self.datafr['span'][idx_data] for num in span]
            tokens_candidates = [token for token in doc if token.is_punct 
                                 and token.text not in tokens_ents 
                                 and token.i+1 not in idxs_error and token.i-1 not in idxs_error] 
            if tokens_candidates:     
                random_tokens = self.__get_k_tokens_candidates(tokens_candidates)  # tokens a tomar en cuenta                
                for token in random_tokens:
                    t_text = self.rules.generate_error_eliminate()
                    self.__set_annotation_errors(idx_data, token.i, token.i+1, token.text, t_text, c.ErrorTag.PUNCT_MISSING.label )
                    self.__update_aux_corrupted_tagged(idx_data, token.i, token.i+1,[t_text], c.ErrorTag.PUNCT_MISSING.label) 
                    self.__update_error_tags_and_tokens(idx_data, token.i, t_text, c.ErrorTag.PUNCT_MISSING.id_num)                    
                    self.datafr['spaces'][idx_data][token.i-1]=False
                    self.datafr['spaces'][idx_data][token.i]=False                                          
                self.__update_version_corrupted(idx_data)
            idx_data += 1



    #crea un arreglo con los indices de inicio y final de los batches para errores
    def __create_idx_batch_dataset(self, num_batches=21):
        """
        Divide el dataset en rangos de índices para procesamiento por lotes.

        Parameters
        ----------
        num_batches : int, opcional
            Número de particiones deseadas. Por defecto es 21

        Returns
        -------
        list of list
            Lista de pares [inicio, fin] para cada lote
        """
        logging.info("Creando batches")
        tamaño_dataset = self.datafr.shape[0]
        tam_batch = tamaño_dataset // num_batches 
        idx_batchs = []
        for i in range(num_batches):
            if i==num_batches-1:
                idx_batchs.append([i*tam_batch, tamaño_dataset])
            else: idx_batchs.append([i*tam_batch, ((i+1)*tam_batch)])
        return idx_batchs
    

    # generar errores en batches , recibe una lista de listas de indices de cada batch [ [idx_inicio_batch, idx_fin_batch],[ii,ij].[ii,ij]]
    def __generate_batches_with_errors(self, idx_batchs): 
        """
        Orquesta la inyección de errores individuales y combinados por batches.

        Asigna cada batch del dataset a un tipo de error específico o a una 
        combinación de múltiples errores 

        Parameters
        ----------
        idx_batchs : list of list
            Índices de inicio y fin para cada bloque del dataset
        """
        logging.info("Generando errores")       
        
        self.apply_errors_ggenre(self.datafr['sentence'][idx_batchs[0][0]:idx_batchs[0][1]], idx_batchs[0][0])
        self.apply_errors_gnumSing(self.datafr['sentence'][idx_batchs[1][0]:idx_batchs[1][1]], idx_batchs[1][0])
        self.apply_errors_gnumPlur(self.datafr['sentence'][idx_batchs[2][0]:idx_batchs[2][1]], idx_batchs[2][0])
       
        self.apply_errors_guart(self.datafr['sentence'][idx_batchs[3][0]:idx_batchs[3][1]], idx_batchs[3][0])
        self.apply_errors_gverbForm(self.datafr['sentence'][idx_batchs[4][0]:idx_batchs[4][1]], idx_batchs[4][0])
        
        self.apply_errors_ggword_order(self.datafr['sentence'][idx_batchs[5][0]:idx_batchs[5][1]], idx_batchs[5][0])
        
        self.apply_errors_stitle(self.datafr['sentence'][idx_batchs[6][0]:idx_batchs[6][1]], idx_batchs[6][0])
        self.apply_errors_saccent(self.datafr['sentence'][idx_batchs[7][0]:idx_batchs[7][1]], idx_batchs[7][0])
        self.apply_errors_punctuation(self.datafr['sentence'][idx_batchs[8][0]:idx_batchs[8][1]], idx_batchs[8][0])
        self.apply_errors_smistake(self.datafr['sentence'][idx_batchs[9][0]:idx_batchs[9][1]], idx_batchs[9][0])
        
        logging.info("Generando errores combinados")
        self.apply_errors_ggenre(self.datafr['sentence'][idx_batchs[10][0]:idx_batchs[10][1]], idx_batchs[10][0])
        self.apply_errors_gverbForm(self.datafr['sentence'][idx_batchs[10][0]:idx_batchs[10][1]], idx_batchs[10][0])
        self.apply_errors_ggword_order(self.datafr['sentence'][idx_batchs[10][0]:idx_batchs[10][1]], idx_batchs[10][0])

        self.apply_errors_gnumSing(self.datafr['sentence'][idx_batchs[11][0]:idx_batchs[11][1]], idx_batchs[11][0])
        self.apply_errors_gverbForm(self.datafr['sentence'][idx_batchs[11][0]:idx_batchs[11][1]], idx_batchs[11][0])

        self.apply_errors_gnumPlur(self.datafr['sentence'][idx_batchs[12][0]:idx_batchs[12][1]], idx_batchs[12][0])
        self.apply_errors_ggword_order(self.datafr['sentence'][idx_batchs[12][0]:idx_batchs[12][1]], idx_batchs[12][0])

        self.apply_errors_smistake(self.datafr['sentence'][idx_batchs[13][0]:idx_batchs[13][1]], idx_batchs[13][0])
        self.apply_errors_gverbForm(self.datafr['sentence'][idx_batchs[13][0]:idx_batchs[13][1]], idx_batchs[13][0])
                
        self.apply_errors_guart(self.datafr['sentence'][idx_batchs[14][0]:idx_batchs[14][1]], idx_batchs[14][0])
        #self.__generate_error_g_verbForm(self.datafr['sentence'][idx_batchs[14][0]:idx_batchs[14][1]], idx_batchs[14][0])
        self.apply_errors_ggword_order(self.datafr['sentence'][idx_batchs[14][0]:idx_batchs[14][1]], idx_batchs[14][0])

        self.apply_errors_punctuation(self.datafr['sentence'][idx_batchs[15][0]:idx_batchs[15][1]], idx_batchs[15][0])
        self.apply_errors_smistake(self.datafr['sentence'][idx_batchs[15][0]:idx_batchs[15][1]], idx_batchs[15][0])

        self.apply_errors_gnumSing(self.datafr['sentence'][idx_batchs[16][0]:idx_batchs[16][1]], idx_batchs[16][0])
        self.apply_errors_smistake(self.datafr['sentence'][idx_batchs[16][0]:idx_batchs[16][1]], idx_batchs[16][0])

        self.apply_errors_gnumPlur(self.datafr['sentence'][idx_batchs[17][0]:idx_batchs[17][1]], idx_batchs[17][0])
        self.apply_errors_smistake(self.datafr['sentence'][idx_batchs[17][0]:idx_batchs[17][1]], idx_batchs[17][0])

        self.apply_errors_guart(self.datafr['sentence'][idx_batchs[18][0]:idx_batchs[18][1]], idx_batchs[18][0])
        self.apply_errors_saccent(self.datafr['sentence'][idx_batchs[18][0]:idx_batchs[18][1]], idx_batchs[18][0])
        self.apply_errors_stitle(self.datafr['sentence'][idx_batchs[18][0]:idx_batchs[18][1]], idx_batchs[18][0])

        self.apply_errors_ggenre(self.datafr['sentence'][idx_batchs[19][0]:idx_batchs[19][1]], idx_batchs[19][0])
        self.apply_errors_gnumSing(self.datafr['sentence'][idx_batchs[19][0]:idx_batchs[19][1]], idx_batchs[19][0])

        self.apply_errors_saccent(self.datafr['sentence'][idx_batchs[20][0]:idx_batchs[20][1]], idx_batchs[20][0])
        self.apply_errors_smistake(self.datafr['sentence'][idx_batchs[20][0]:idx_batchs[20][1]], idx_batchs[20][0])


    
    #limpiar dataset
    def clean_datafr(self):
        """
        Elimina filas del DataFrame que contienen valores nulos o vacíos

        Returns
        -------
        pandas.DataFrame
            DataFrame limpio y sin registros incompletos.
        """
        df_no_empty = self.datafr[~self.datafr.map(lambda x: x is None or x == '' or (isinstance(x, list) and not x)).any(axis=1)]
        return df_no_empty 
    
    
    def inyect_data_errors(self):
        """
        Punto de entrada principal para generar el dataset corrupto completo.

        Crea los lotes, inyecta los errores, limpia los datos nulos y 
        realiza un barajado (shuffle) final del dataset.

        Returns
        -------
        pandas.DataFrame
            El nuevo dataset final procesado y listo para entrenamiento
        """
        logging.info("Creando nuevo conjunto de datos")
        idx_batchs = self.__create_idx_batch_dataset(21) #crear 19 batches (10 errores individuales) 6 para errores combinados: gwo, guart, gverbForm 
        
        self.__generate_batches_with_errors(idx_batchs)
        logging.info("Limpiando nuevo conjunto de datos")

        df_cleaned = self.clean_datafr()
        df_cleaned = df_cleaned.sample(frac=1, random_state=42).reset_index(drop=True)
        return df_cleaned
    

