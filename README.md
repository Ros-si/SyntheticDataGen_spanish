# Synthetic Data Generator for Spanish GEC (Grammatical Error Correction)

Este módulo es el motor de generación de datos sintéticos del proyecto principal. Su propósito fundamental es mitigar la escasez de datos de corpus paralelos para la tarea de Corrección de Errores Gramaticales en el idioma español, mediante la inyección probabilística y controlada de ruido lingüístico sobre textos limpios.

Se utiliza un analizador gramatical para la creación de errores realistas de concordancia y sintaxis que emulan las fallas comunes de hablantes nativos y aprendices de una segunda lengua (L2).

## Características Clave
* **Análisis Gramatical con Reglas:** Identificación precisa de sustantivos, adjetivos y verbos para la manipulación morfológica estructurada.
* **Preparación Avanzada para Modelos de Corrección y Detección:** El output no solo sirve para tareas *seq2seq* de corrección, sino que genera etiquetas de mapeo de tokens alineadas para entrenar modelos auxiliares de clasificación/detección de errores (Token Classification).

---

## 📊 Categorización de Errores

Los tipos de errores que se generan son los siguientes:

### Tipos de Errores Implementados

| Categoría | Etiqueta | Tipo de Error / Descripción |
| :--- | :--- | :--- |
| **Gramaticales** | `G-gen` | Uso incorrecto del género en: artículos y pronombres. |
| | `G-nSing` | Uso incorrecto del número singular en sustantivos. |
| | `G-nPlur` | Uso incorrecto del número plural en sustantivos. |
| | `G-verbForm` | Uso incorrecto de la forma del verbo y de verbos auxiliares. |
| | `G-uArt` | Ausencia de artículos. |
| | `G-wo` | Alteración del orden sintáctico de la oración (*Word Order*). |
| **Ortográficos** | `S-title` | Uso incorrecto de letras mayúsculas al comienzo de las palabras. |
| | `S-noAccent` | Ausencia de tildes (acentuación gráfica). |
| | `S-mistake` | Error de escritura, uso incorrecto de letras (ortografía fonética). |
| **Puntuación** | `P-missing` | Ausencia de signos de puntuación. |

---

## Estructura del Conjunto de Datos Corrupto

Las anotaciones presentan información complementaria a la corrección del texto, con el propósito de que puedan emplearse para verificar la posición y el tipo de error presente. Además, esto faculta al dataset para ser empleado en el entrenamiento de otros modelos orientados a la **detección de errores**.

La estructura del del dataset generado adopta el siguiente formato:

| Campo | Tipo / Formato | Descripción |
| :--- | :--- | :--- |
| `sentence` | Texto (`str`) | Oración original, gramaticalmente correcta. |
| `corrupted` | Texto (`str`) | Versión paralela de la oración que contiene los errores inyectados. |
| `tokens` | Lista (`list`) | Arreglo de los tokens correspondientes a la oración con errores (`corrupted`). |
| `error_tags` | Lista (`list`) | Arreglo codificado numéricamente que indica las posiciones de los tokens que presentan algún error. |
| `error_type` | Lista (`list`) | Arreglo que especifica las cadenas textuales de los tipos de errores presentes en la oración. |
| `span` | Lista de tuplas | Arreglo de duplas que especifican los índices de caracteres del texto corrupto en la oración `(idx_inicio, idx_fin)`. |
| `annotation` | Lista de tuplas | Arreglo de duplas que indican la versión del texto con error y su contraparte correcta `(texto_corrupto, texto_correcto)`. |
| `corrupted_tagged` | Texto (`str`) | Oración con errores, donde cada término erróneo se encuentra marcado inline con su tipo de error correspondiente bajo la estructura: `<TipoDeError errorPresente>`. |

### 🔍 Ejemplos Prácticos del Dataset

A modo de ilustración, la siguiente tabla muestra cómo se estructuran y mapean internamente dos instancias procesadas por el pipeline de corrupción:

| Atributo / Campo | Ejemplo 1 (Error de Género) | Ejemplo 2 (Error de Número Singular) |
| :--- | :--- | :--- |
| `sentence` | Futbolistas del Club Atlético Peñarol en los años 1940 | Varios emuladores para los dispositivos de entrada y salida. |
| `corrupted` | Futbolistas del Club Atlético Peñarol en las años 1940 | Varios emulador para los dispositivo de entrada y salida. |
| `tokens` | `['Futbolistas', 'del', 'Club', 'Atlético', 'Peñarol', 'en', 'las', 'años', '1940']` | `['Varios', 'emulador', 'para', 'los', 'dispositivo', 'de', 'entrada', 'y', 'salida', '.']` |
| `error_tags` | `[0, 0, 0, 0, 0, 0, 1, 0, 0]` | `[0, 2, 0, 0, 2, 0, 0, 0, 0, 0]` |
| `error_type` | `["G-gen"]` | `["G-nSing", "G-nSing"]` |
| `span` | `[(6,7)]` | `[(1,2),(4,5)]` |
| `annotation` | `[('las', 'los')]` | `[('emulador', 'emuladores'), ('dispositivo', 'dispositivos')]` |
| `corrupted_tagged` | Futbolistas del Club Atlético Peñarol en `<G-gen las>` años 1940 | Varios `<G-nSing emulador>` para los `<G-nSing dispositivo>` de entrada y salida. |

### Identificación Numérica de Errores (`error_tags`)

Para facilitar el entrenamiento de tareas de clasificación de tokens, los identificadores numéricos asignados dentro del arreglo `error_tags` corresponden a la siguiente matriz de mapeo:

| Id del Error | Tipo de Error Correspondiente |
| :---: | :--- |
| **0** | Ningún error (Token correcto) |
| **1** | Gramatical: género (`G-gen`) |
| **2** | Gramatical: número singular (`G-nSing`) |
| **3** | Gramatical: número plural (`G-nPlur`) |
| **4** | Gramatical: forma verbal (`G-verbForm`) |
| **5** | Gramatical: artículo (`G-uArt`) |
| **6** | Gramatical: orden de palabras (`G-wo`) |
| **7** | Puntuación: signo omitido (`P-missing`) |
| **8** | Ortográfico: mayúscula inicial (`S-title`) |
| **9** | Ortográfico: acento omitido (`S-noAccent`) |
| **10** | Ortográfico: error ortográfico fonético (`S-mistake`) |

---
## Requisitos e Instalación

Este módulo está desarrollado y probado en entornos de **Python 3.13**. Sigue estos pasos para configurar tu entorno local:

1. **Clonar el repositorio:**
   ```bash
   git clone [https://github.com/Ros-si/SyntheticDataGen_spanish.git](https://github.com/Ros-si/SyntheticDataGen_spanish.git)
   cd SyntheticDataGen_spanish
    ```

2. **Instalar las dependencias:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Autenticación en Hugging Face:**
Para permitir que el script guarde de forma automática el dataset generado directamente en tu perfil del Hub de Hugging Face, inicia sesión en la CLI:
   ```bash
   huggingface-cli login
   ```
## Generación del Nuevo Conjunto de Datos
El archivo main.py actúa como el orquestador del generador. El comportamiento del algoritmo de inyección se parametriza a través del diccionario config dentro del script.

### Parámetros de Configuración

| Clave / Parámetro | Tipo | Descripción |
| :--- | :---: | :--- |
| `"sampling"` | `float` | **Tamaño de muestra:** Muestra de registros que se extraerán del conjunto de datos original para ser procesadas (rango entre `0.0` y `1.0`). |
| `"min_string"` | `int` | **Longitud mínima de palabras:** Filtro que define la cantidad mínima de palabras que debe tener una secuencia. |
| `"max_string"` | `int` | **Longitud máxima de palabras:** Filtro que limita la cantidad máxima de palabras por secuencia. |
| `"name_dataset"` | `str` | **Nombre del nuevo dataset:** Identificador o nombre que recibirá el conjunto de datos resultante. |
| `"path_data"` | `str` | **Ruta de almacenamiento local:** Dirección física en el disco donde se guardará la copia de seguridad del dataset generado en formato `.csv`. |
| `"validation_size"`| `float` | **Tamaño del split de validación:** Proporción de los datos para el subconjunto de validación (rango entre `0.0` y `1.0`). |
| `"test_size"` | `float` | **Tamaño del split de prueba:** Proporción de los datos para el subconjunto de prueba/test (rango entre `0.0` y `1.0`). |
| `"nlp"` | `object`| **Instancia de spaCy:** Pipeline del modelo de lenguaje en español (`es_core_news_sm`) encargado de realizar el análisis morfosintáctico. |
| `"data_source"` | `str` | **Dataset de origen:** Corpus/Dataset base, sobre el cual se realizará la inyección de errores. |
| `"column_source"` | `str` | **Columna objetivo:** Nombre exacto de la columna dentro del dataset de origen que almacena el texto que va a recibir la inyección de errores. |

**Ejemplo de configuración**:
    ´´´bash
    nlp = spacy.load("es_core_news_md")
    dataHF = load_dataset("wikimedia/wikipedia", "20231101.es", split="train")
    column_name = "text"

    config = {
        "sampling": 0.1,
        "min_string": 6,
        "max_string": 128,
        "name_dataset": "new_dataset_to_GEC-es",
        "path_data": "./data/",
        "validation_size": 0.05,
        "test_size": 0.05,
        "nlp": nlp,
        "data_source":dataHF,
        "column_source":column_name
    }
    ´´´ 

### Ejecución del Pipeline

Una vez que configuradas las variables en el diccionario config dentro de main.py, se puede iniciar el proceso completo (ingesta, inyección probabilística de errores, segmentación de conjuntos de datos y exportación) ejecutando el script como un módulo desde la raíz del repositorio:
    ´´´bash
    python -m main
    ´´´
**Nota**
Es necesario la previa configuración de credenciales de Hugging Face (huggingface-cli login), ya que se realizará el push de manera directa al repositorio en el Hub.
