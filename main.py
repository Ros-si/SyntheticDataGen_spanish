from src.datasetGenerator import DatasetGenerator
import spacy
from datasets import load_dataset

def main():
    # Parámetros de configuración
    nlp = spacy.load("es_core_news_md")
    dataHF = load_dataset("wikimedia/wikipedia", "20231101.es", split="train")
    column_name = "text"
    config = {
        "sampling": 0.0070,
        "min_string": 6,
        "max_string": 128,
        "name_dataset": "WikiCorrupted_spanish_to_GEC-GED_medium",
        "path_data": "./data/",
        "validation_size": 0.015,
        "test_size": 0.015,
        "nlp": nlp,
        "data_source":dataHF,
        "column_source":column_name
    }
    
    # Instanciación y ejecución
    gen_data = DatasetGenerator(**config)
    gen_data.run_pipeline()

if __name__ == "__main__":
    main()