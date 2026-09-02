from src.datasetGenerator import DatasetGenerator
import spacy
from datasets import load_dataset

def main():
    # Parámetros de configuración
    nlp = spacy.load("es_core_news_md")
    dataHF = load_dataset("wikimedia/wikipedia", "20231101.es", split="train")
    column_name = "text"
    error_rate = 5 
    config = {
        "sampling": 0.025,
        "min_string": 6,
        "max_string": 128,
        "name_dataset": "WikiCorrupted_spanish_to_GEC-GED_L",
        "path_data": "./data/",
        "validation_size": 0.005,
        "test_size": 0.005,
        "nlp": nlp,
        "data_source":dataHF,
        "column_source":column_name
    }
    
    # Instanciación y ejecución
    gen_data = DatasetGenerator(**config)
    gen_data.run_pipeline(error_rate)

if __name__ == "__main__":
    main()