# Código principal
En esta carpeta se encuentran los archivos necesarios para obtener predicciones y segmentaciones.


## Modelos
Las clases de los modelos existentes se encuentran en el archivo [models.py](models.py):
- EfficientNet_2CLS: Clase creada para el modelo EfficientNet B6 reentrenado. Se pueden encontrar los archivos pt de los modelos por cada preprocesamiento en Kaggle ([EfficientNetB6_rgi_stack](https://www.kaggle.com/models/domenicomoralesortiz/efficientnetb6-rgi-stack), [EfficientNetB6_rgi_unsharp_mask](https://www.kaggle.com/models/domenicomoralesortiz/efficientnetb6-rgi-unsharp-mask))
- DINO_2CLS: Clase creada para el modelo DINOv2 ViT Base reentrenado. Se pueden encontrar los archivos pt de los modelos por cada preprocesamiento en Kaggle ([DINOv2ViTBase_rgi_stack](https://www.kaggle.com/models/domenicomoralesortiz/dinov2vitbase-rgi-stack), [DINOv2ViTBase_rgi_unsharp_mask](https://www.kaggle.com/models/domenicomoralesortiz/dinov2vitbase-rgi-unsharp-mask))


## Funciones
Las funciones para utilizar los modelos, crear carpetas, usar transformaciones, crear segmentaciones y visualizar resultados se encuentran en el archivo [helpers.py](helpers.py)

## Libreta principal
El Jupyter Notebook [MainNotebook.ipynb](src/MainNotebook.ipynb) es una libreta interactiva creada para ingresar los parámetros y obtener resultados solamente con ejecutar la libreta completa.

Los parámetros a modificar se encuentran en la primera celda y se describen en las siguientes secciones.

### Parámetros de ambiente
- **COLAB** (bool): indicar con un valor booleano si se está ejecutando en Google Colab.
- **NOTEBOOK_PATH** (str): indicar la ruta de la carpeta donde el notebook se encuentra en Google Drive.

### Parámetros de clasificación
- **CORRER_CLASIFICACION** (bool): indicar con un valor booleano si se hará inferencia de clasificación.
- **MODELO** (str): indicar el modelo a usar de las opciones 'DINOv2ViTBase' o 'EfficientNetB6'
- **PREPROCESAMIENTO** (str): indicar el preprocesamiento de las opciones 'rgi_stack' o 'rgi_unsharp_mask'
- **RUTA_FITS** (str): indicar la ruta de la carpeta donde se encuentren las imágenes fits a inferenciar.
- **RUTA_MODELOS** (str): indicar la ruta de la carpeta donde se encuentran los modelos.
- **FORMATO_GUARDADO** (str): indicar el formato en el que las imágenes clasificada se van a guardar de las opciones 'fits', 'png' o 'all'
- **THRESHOLD** (float): indicar el umbral a utilizar para clasificar la imágen, los valores recomendados son 0.7 para DINOv2ViTBase_rgi_stack, 0.68 para DINOv2ViTBase_rgi_unsharp_mask, 0.37 para EfficientNetB6_rgi_Stack y 0.86 para EfficientNetB6_rgi_unsharp_mask.

### Parámetros de segmentación
- **CORRER_SEGMENTACION** (bool): indicar con un valor booleano si se hará inferencia de segmentación de anillos.
- **RUTA_ARCHIVO_FITS** (str): indicar la ruta del archivo fits a inferenciar.
- **TIPO_ANILLO** (str): indicar el tipo de anillo que se encuentra en la imágen de las opciones 'inner', 'outer' o 'inner+outer'
- **MOSTRAR_MASCARA** (bool): indicar con un valor booleano si se desea mostrar la máscara de detección en la segmentación.
- **VISUALIZAR_SEGMENTACION** (bool): indicar con un valor booleano si se desea visualizar la imagen después de la segmentación.