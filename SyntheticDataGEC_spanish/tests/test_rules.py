"""Sample unit test """
# pylint: disable=unused-variable,expression-not-assigned,singleton-comparison
import pytest
import spacy
import random
from unittest.mock import patch
from SyntheticDataGEC_spanish.rules import RulesHandler

@pytest.fixture(scope="module")
def nlp():
    nlp_model = spacy.load("es_core_news_lg")    
    return nlp_model


@pytest.fixture
def generator(nlp):
    return RulesHandler(nlp=nlp)


# Tests modificación de género 
@pytest.mark.parametrize("word, expected", [
    ("el", "la"),  # Masculino -> Femenino
    ("la", "el"),  # Femenino -> Masculino
    ("Aquel", "Aquella"),  # Masculino -> Femenino
    ("Ellos", "Ellas"),  # Masculino -> Femenino
    ("ningun", "ninguna"),  # Masculino -> Femenino
    ("ella", "él"),   # Femenino -> Masculino
    ("esa", "ese"), # Femenino -> Masculino
    ("nosotras", "nosotros"), # Femenino -> Masculino
    ("del", "de la"), # Preposición + artículo masculino -> femenino
])
def test_generate_error_genre(nlp, generator, word, expected):
    doc = nlp(word)
    assert generator.generate_error_genre(doc[0]) == expected


# Tests modificacion a número plural
@pytest.mark.parametrize("word, expected", [
    ("vaca", "vacas"),      # Vocal + s
    ("luz", "luces"),        # z -> ces
    ("Reloj", "Relojes"),    # Consonante + es
    ("dux", "dux"),          # Invariable
    ("compás", "compases"),  # Aguda s -> es (sin tilde)
])
def test_generate_error_gnumPlur(nlp, generator, word, expected):
    doc = nlp(word)
    assert generator.generate_error_gnumPlur(doc[0]) == expected


# --- Tests de Lematización (Singular y Verbos) ---
@pytest.mark.parametrize("word, expected", [
    ("vacas", "vaca"),      # plural -> singular
    ("Estaciones", "Estación"),        # plural -> singular
    ("Jugamos", "Jugar"),    # 3persona plural -> Infinitivo
    ("Comiendo", "Comer"),  #
    ("tenemos", "tener"), # 1persona, plural -> infinitivo
    ("cantando", "cantar"), # Gerundio -> Infinitivo
    ("Estaba", "Estar"),  # 1persona singular pretérito imperfecto -> cambio de raíz (infinitivo)
    ("voy", "ir"), # 1persona singular presente -> cambio de raíz (infinitivo)
    ("fuimos", "ir"), # 1persona plural pretérito -> cambio de raíz (infinitivo)
    ("caminar", ""), # Infinitivo -> ""(sin cambio)
])
def test_generate_error_lemma(nlp, generator, word, expected):
    doc = nlp(word)
    assert generator.generate_error_lemma(doc[0]) == expected


# Tests: modificación de Orden de Palabras
@pytest.mark.parametrize("input_texts, expected", [
    (["come", "manzanas"], ["manzanas", "come"]),  # [{'POS':'VERB'}, {'POS':'NOUN'}]  
    (["corre", "rápido"], ["rápido", "corre"]),  # [{'POS':'VERB'}, {'POS':'ADV'}] 
    (["casa", "grande"], ["grande", "casa"]),   # [{'POS':'NOUN'}, {'POS': 'ADJ'}] 
    (["hermosa", "vista"], ["vista", "hermosa"]),   # [{'POS':'ADJ'}, {'POS': 'NOUN'}] 
    (["es", "amado"], ["amado", "es"]),   # [{'POS': 'AUX', 'LEMMA': 'ser'}, {'POS': 'VERB'}] 
    (["está", "feliz"], ["feliz", "está"]),  # [{'POS': 'AUX', 'LEMMA': 'estar'}, {'POS': 'ADJ'}] 
    
    # Casos con Mayúsculas (Title Case al inicio)
    (["El", "perro"], ["perro", "el"]),
    (["Estaba", "cansado"], ["cansado", "estaba"]),
])
def test_generate_error_ggword_order(generator, input_texts, expected):
    assert generator.generate_error_ggword_order(input_texts) == expected
    

# Tests: modificación de estilo -> Title
def test_generate_error_title(nlp, generator):
    assert generator.generate_error_title(nlp("hola")[0]) == "Hola"
    assert generator.generate_error_title(nlp("Mundo")[0]) == ""


def test_generate_error_accent(nlp, generator):
    doc = nlp("camión águila")
    assert generator.generate_error_accent(doc[0]) == "camion"
    assert generator.generate_error_accent(doc[1]) == "aguila"


# Tests: modificacion de letras -> Errores de ortografía 
def test_generate_spelling_mistake(nlp, generator):
    doc = nlp("vaca")
    # Forzamos a random.choice a elegir la 'v' para que el test sea determinista
    with patch('random.choice', return_value='v'):
        result = generator.generate_spelling_mistake(doc[0])
        assert result == "baca"


