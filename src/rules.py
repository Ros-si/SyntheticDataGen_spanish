import spacy
import random
import re
from unidecode import unidecode
from spacy.matcher import Matcher
from spacy.tokens import Doc
import silabeador

from . import constants as c

class RulesHandler:
    """
    Clase encargada de la inyección de errores.
    Utiliza spaCy y reglas lingüísticas personalizadas para transformar los tokens
    """
    def __init__(self, nlp):
        """
        Inicializa el gestor de reglas
        
        Parameters
        ----------
        nlp : spacy.lang.Language
            Pipeline de spaCy cargado, utilizado para el vocabulario y metadatos
        matcher : spacy.matcher.Matcher
            Instancia de Matcher para definir patrones de detección de errores
        """
        self.nlp = nlp
        self.matcher = Matcher(nlp.vocab)

        
    def generate_error_genre(self, token):
        """
        Cambia el género de artículos y pronombres, modifica el token con el token
        del género opuesto
        
        Parameters
        ----------
        token: spacy.Token
            Token de entrada que será transformado
        Returns
        -------
        str
            El texto del token con el género invertido (masculino a femenino o viceversa)
        """
        actual_token = token.text.lower()
        if actual_token in c.ART_PRON_MASC:
            t_text = c.ART_PRON_FEM[c.ART_PRON_MASC.index(actual_token)] 
        elif actual_token in c.ART_PRON_FEM:
            t_text = c.ART_PRON_MASC[c.ART_PRON_FEM.index(actual_token)]
        if token.is_title: 
            t_text = t_text.title()  
        return t_text
    
    #palabras que terminan en -x o -s , se agrega "es",
    # permanecen invariables los polisílabos agudos cuando se trata de voces compuestas cuyo segundo elemento es ya un plural: ciempiés, pl. ciempiés (no ⊗‍ciempieses); buscapiés, pl. buscapiés (no ⊗‍buscapieses); pasapurés, pl. pasapurés (no ⊗‍pasapureses).
    #si es aguda -> su plural se forma añadiendo -es y quitando la tilde (porque deja de ser aguda).
    #si es monosilaba o polisilaba se aañade "es"
    def __plural_x_s_es(self, token):
        """
        Aplica reglas de pluralización para palabras terminadas en -x o -s.
        Gestiona excepciones de palabras invariables y polisílabas agudas.
        
        Parameters
        ----------
        token : str
            Cadena de texto del token.

        Returns
        -------
        str
            Forma pluralizada del token según las reglas definidas.
        """
        invariables=['dux','pasapurés','unisex','beis','relax']
        new_token=token
        if token in invariables: 
            return new_token    
        elif token[-3:] == 'iés':
            return new_token
        else:
            syllables = silabeador.Syllabification(token)
            n=len(syllables.syllables)
            if syllables.stress ==-1: #es aguda
                new_token = unidecode(token)
                return new_token+'es'
            elif n>3 or n ==1:
                return new_token+'es'        
        return new_token
    
    #Sustantivos y adjetivos terminados en -l, -r, -n, -d, -z, -j. 
    #para palabras que finalizen en z -> ces
    #Si no van precedidas de otra consonante, forman el plural con -es:
    #dócil, pl. dóciles; color, pl. colores; pan, pl. panes; césped, pl. céspedes; cáliz, pl. cálices; reloj, pl. relojes. Los extranjerismos que terminen en estas consonantes deben seguir esta misma regla: píxel, pl. píxeles; máster, pl. másteres; pin, pl. pines; interfaz, pl. interfaces; sij, pl. sijes.
    #El plural de las palabras esdrújulas es invariable: el cárdigan/los cárdigan, el mánager/los mánager (también el mánayer/los mánayer), el trávelin/los trávelin.
    def __plural_lrndzj_es(self, token):
        """
        Aplica reglas de pluralización para palabras terminadas en -l, -r, -n, -d, -z, -j

        Parameters
        ----------
        token : str
            Cadena de texto del token

        Returns
        -------
        str
            Forma pluralizada del token según las reglas definidas
        """
        new_token=token
        if token[-1]=='z':
            return new_token[:-1]+'ces'
        elif token[-1] in ['l','r','n','d','j'] and c.PATTERN_ENDVOCAL.match(token[:-1]):
            syllables = silabeador.Syllabification(token)
            if syllables.stress ==-3: #esdrújulas-> invariable
                return new_token
            else:return new_token+'es'
        return token
    
    def generate_error_gnumPlur(self, token):
        """
        Transforma un token singular a su forma plural para inducir un error de número.

        Parameters
        ----------
        token : spacy.tokens.Token
            Token de entrada

        Returns
        -------
        str
            Cadena de texto con el plural generado
        """
        if c.PATTERN_ENDVOCAL.match(token.text): #si termina en vocal -> +s
            t_text =token.text
            t_text = t_text+'s'
        elif token.text[-1] in ['x','s']: #si termina en -vocal + x|s -> +es
                t_text = self.__plural_x_s_es(token.text)
        elif token.text[-1] in ['z','l','r','n','d','j'] and c.PATTERN_ENDVOCAL.match(token.text[:-1]): # -vocal + z|l|r|n|d|j  -> +es
            t_text = self.__plural_lrndzj_es(token.text)
        else: t_text = ""
        if token.is_title : t_text=t_text.title()
        return t_text
    
    ##para gnumsing y verbform
    def generate_error_lemma(self, token):
        """
        Reemplaza un token por su lema base para inducir errores de concordancia.

        Parameters
        ----------
        token : spacy.tokens.Token
            Token de entrada a lematizar

        Returns
        -------
        str
            Texto lematizado o cadena vacía si ya estaba en su forma base
        """
        t_lemma = token.lemma_
        if t_lemma != token.text:
            if token.is_title: 
                t_lemma=t_lemma.title()
        else: 
            return ""
        return t_lemma
            
    
    ##para errores guart, y punctuation
    def generate_error_eliminate(self):
        """
        Devuelve un espacio vacío para simular la eliminación de un token.

        Returns
        -------
        str
            Un carácter de espacio en blanco.
        """
        return " "
    
    def generate_error_ggword_order(self, texts):
        """
        Invierte el orden de los elementos en una secuencia

        Parameters
        ----------
        texts : list of str
            Lista de palabras a reordenar

        Returns
        -------
        list of str
            Lista de entrada en orden inverso
        """
        if texts[0].istitle(): 
            texts[0] = texts[0].lower()
        text_reverse = texts[::-1]
        return text_reverse

    def generate_error_title(self, token):
        """
        Aplica mayúscula inicial a un token para inducir errores de capitalización

        Parameters
        ----------
        token : spacy.tokens.Token
            Token de entrada

        Returns
        -------
        str
            Texto del token con la primera letra en mayúscula
        """
        word_title = token.text.title() 
        if word_title == token.text:
            return "" #token.text.lower()
        return word_title

    def generate_error_accent(self, token):
        """
        Elimina los acentos y diacríticos del texto del token

        Parameters
        ----------
        token : spacy.tokens.Token
            Token de entrada

        Returns
        -------
        str
            Texto del token sin tildes 
        """
        word = unidecode(token.text)
        if token.is_title: 
            word=word.title()
        return word
    
    def generate_spelling_mistake(self, token):
        """
        Induce un error ortográfico fonético aleatorio mediante el reemplazo de caracteres

        Parameters
        ----------
        token : spacy.tokens.Token
            Token de entrada a modificar

        Returns
        -------
        str
            Texto del token con una sustitución fonética aplicada
        """
        t_text = token.text
        possible_keys = [key for key in c.PHONETIC_REPLACEMENTS if key in t_text]
        selected_key = random.choice(possible_keys)
        t_text= t_text.replace(selected_key, c.PHONETIC_REPLACEMENTS[selected_key], 1)
        if token.is_title: 
            t_text=t_text.title()
        return t_text
    
   