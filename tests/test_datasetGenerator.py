import pytest
import pandas as pd
import unicodedata
from src.datasetGenerator import DatasetGenerator 

class TestDatasetGenerator:

    @pytest.fixture
    def generator(self):
        """Instancia base para los tests."""
        return DatasetGenerator(
            path_data="./data/",
            sampling=0.0001,
            min_string=3,
            max_string=10,
            validation_size=0.2,
            test_size=0.2
        )

    def test_normalize_sentence(self, generator):
        """Verifica que la normalización NFKC funcione (ej. diéresis o caracteres combinados)."""
        texto_sucio = "esta\u0301"
        normalizado = generator._DatasetGenerator__normalize_sentence(texto_sucio)
        print(texto_sucio)
        assert normalizado == "está"

    def test_divide_sentences(self, generator):
        """Prueba la división por saltos de línea simples y dobles."""
        text = "Oración 1\nOración 2\n\nOración 3"
        # Acceso al método mangled
        result = generator._DatasetGenerator__divide_sentences(text)
        assert len(result) == 3
        assert "Oración 2" in result

    def test_filter_sentences_length(self, generator):
        """Verifica que filtre oraciones fuera del rango de palabras (min_string=3, max_string=10)."""
        sentences = [
            "Dos palabras",                  # Muy corta (2) -> Fuera
            "Tres palabras exactas",         # OK (3) -> Dentro
            "Esta es una oración perfecta",   # OK (5) -> Dentro
            "Esta es una oración que definitivamente tiene más de diez palabras en total" # Muy larga -> Fuera
        ]
        filtered = generator._DatasetGenerator__filter_sentences(sentences)
        assert len(filtered) == 2
        assert sentences[1] in filtered
        assert sentences[2] in filtered

    def test_filter_sentences_exclude_patterns(self, generator):
        """Verifica que elimine URLs y rutas de archivos."""
        sentences = [
            "Visita https://google.com para más", # URL -> Fuera
            "El archivo está en /usr/bin/python",  # Ruta -> Fuera
            "Oración limpia y válida"              # OK -> Dentro
        ]
        filtered = generator._DatasetGenerator__filter_sentences(sentences)
        assert len(filtered) == 1
        assert "Oración limpia y válida" in filtered

    def test_prepare_features(self, generator):
        """Verifica que el DataFrame resultante tenga todas las columnas requeridas."""
        flat_list = ["Oración uno", "Oración dos"]
        df = generator._DatasetGenerator__prepare_features(flat_list)
        
        expected_columns = [
            'sentence', 'corrupted', 'tokens', 'error_tags', 
            'error_type', 'span', 'annotation', 'corrupted_tagged', 
            'aux_corrupted_tagged', 'spaces'
        ]
        assert all(col in df.columns for col in expected_columns)
        assert len(df) == 2
        assert isinstance(df['tokens'][0], list)

    def test_generate_splits(self, generator):
        """Verifica que las proporciones de los splits sean correctas."""
        # Creamos un DF de prueba con 10 filas
        data = {'sentence': [f"S{i}" for i in range(10)]}
        df = pd.DataFrame(data)
        
        # validation=0.2, test=0.2 -> train=0.6
        splits = generator._DatasetGenerator__generate_splits(df)
        
        assert len(splits['test']) == 2
        assert len(splits['validation']) == 2
        assert len(splits['train']) == 6

