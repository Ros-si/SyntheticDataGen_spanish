from enum import Enum
import re

class ErrorTag(Enum):
    # Definición: (ID numérico, Etiqueta para el dataset)
    NO_ERROR = (0, "Correct")
    GENDER = (1, "G-gen")
    NUM_SING = (2, "G-nSing")
    NUM_PLUR = (3, "G-nPlur")
    VERB_FORM = (4, "G-verbForm")
    ART_MISSING = (5, "G-uArt")
    WORD_ORDER = (6, "G-wo")
    PUNCT_MISSING = (7, "P-missing")
    TITLE = (8, "S-title")
    ACCENT = (9, "S-noAccent")
    SPELLING = (10, "S-mistake")

    def __init__(self, id_num, label):
        self.id_num = id_num
        self.label = label
   
# Pronombres y articulos de género masculino y femenino
ART_PRON_MASC = ['él', 'nosotros', 'vosotros', 'ellos','un','unos','ese','esos','aquel','aquellos','los','el','este','estos','del','algun', 'algunos','ningun','todos','todo']
ART_PRON_FEM = ['ella', 'nosotras', 'vosotras','ellas','una','unas','esa','esas','aquella','aquellas','las','la','esta','estas','de la','alguna','algunas','ninguna','todas','toda']

PATTERN_ENDVOCAL = re.compile(r'\b\w*[aeiouáéíóúü]\b', re.IGNORECASE)
PATTERN_ACCENT = re.compile(r'\b\w*[áéíóú]\w*\b', re.IGNORECASE)
#special_punctuation =['(','[','{','"','¡','¿','"']
PHONETIC_REPLACEMENTS = {"h":"", "s":"z", "z":"s", "v":"b", "b":"v","m":"n","n":"m", "y":"ll", "ll":"y","je":"ge","ge":"je","gi":"ji","gi":"ji"}

# patterns_wo: lista que define los patrones de palabras en términos de sus categorías gramaticales (POS: Part-of-Speech) que serán detectados en las oraciones
PATTERNS_WO=[[{'POS':'VERB'}, {'POS':'NOUN'}], # combinaciones de verbo seguido de sustantivo
            [{'POS':'VERB'}, {'POS':'ADV'}], # combinaciones de verbo seguido de un adverbio
            [{'POS':'NOUN'}, {'POS': 'ADJ'}], # combinaciones de un sustantivo seguido de adjetivo
            [{'POS':'ADJ'}, {'POS': 'NOUN'}], # combinaciones de adjetivo seguido de sustantivo
            [{'POS': 'AUX', 'LEMMA': 'ser'}, {'POS': 'VERB'}], # "ser" como verbo auxiliar seguido de otro verbo
            [{'POS': 'AUX', 'LEMMA': 'estar'}, {'POS': 'ADJ'}], # "estar" como verbo auxiliar seguido de un adjetivo
            [{'POS': 'AUX', 'LEMMA': 'estar'}, {'POS': 'VERB'}], # "estar" como verbo auxiliar seguido de un verbo
            [{'POS': 'AUX', 'LEMMA': 'ser'}, {'POS': 'ADJ'}], #  "ser" como verbo auxiliar seguido de un adjetivo
            [{'POS':'VERB'}, {'POS':'ADJ'}]
            ]