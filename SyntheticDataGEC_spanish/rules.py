import spacy
import random
import re
from unidecode import unidecode
from spacy.matcher import Matcher
from spacy.tokens import Doc
import silabeador

from . import constants as c

class RulesHandler:
    def __init__(self, nlp):
        self.nlp = nlp
        self.matcher = Matcher(nlp.vocab)

        
    def generate_error_genre(self, token):
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
        print(token.pos_, token.lemma_)
        t_lemma = token.lemma_
        if t_lemma != token.text:
            if token.is_title: 
                t_lemma=t_lemma.title()
        else: return ""
        return t_lemma
            
    
    ##para errores guart, y punctuation
    def generate_error_eliminate(self):
        return " "
    
    def generate_error_ggword_order(self, texts):
        if texts[0].istitle(): 
            texts[0] = texts[0].lower()
        text_reverse = texts[::-1]
        return text_reverse

    def generate_error_title(self, token):
        word_title = token.text.title() 
        if word_title == token.text:
            return "" #token.text.lower()
        return word_title

    def generate_error_accent(self, token):
        word = unidecode(token.text)
        if token.is_title: 
            word=word.title()
        return word
    
    def generate_spelling_mistake(self, token):
        t_text = token.text
        possible_keys = [key for key in c.PHONETIC_REPLACEMENTS if key in t_text]
        selected_key = random.choice(possible_keys)
        t_text= t_text.replace(selected_key, c.PHONETIC_REPLACEMENTS[selected_key], 1)
        if token.is_title: 
            t_text=t_text.title()
        return t_text
    
   