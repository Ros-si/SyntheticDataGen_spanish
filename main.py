from src.datasetGenerator import DatasetGenerator
import spacy

def main():
    # Parámetros de configuración
    nlp = spacy.load("es_core_news_lg")
    config = {
        "sampling": 0.00035,
        "min_string": 8,
        "max_string": 120,
        "name_dataset": "WikiCorrupted_spanish_to_GEC-GED_toTEST2",
        "path_data": "./data/",
        "validation_size": 0.025,
        "test_size": 0.025,
        "nlp": nlp
    }
    
    # Instanciación y ejecución
    gen_data = DatasetGenerator(**config)
    gen_data.run_pipeline()

if __name__ == "__main__":
    main()