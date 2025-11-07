# ITESM-RingGalaxyClassification
Este repositorio contiene un proyecto realizado por alumnes del ITESM de la maestría de Inteligencia Artificial en colaboración con el Instituto de Astronomía de la UNAM.

El proyecto se divide en dos secciones:
- Clasificación de galaxias (anillo / no anillo)
- Segmentación de anillos según su tipo (interno, externo o interno+externo)

## Configuración de imágenes
Las imágenes utilizadas para entrenar los modelos en este proyecto corresponen a [DESI Legacy Survey](https://www.legacysurvey.org/) en la versión DR10.

Los archivos FITS de entrenamiento e inferencia deben tener la siguiente configuración:
- Escala: 0.262
- Tamaño de imagen: 224x224
- Bandas: gri

## Inicio rápido
La manera más rápida de inferir y obtener predicciones y segmentaciones es utilizar el Jupyter Notebook principal en Google Colaboratory, ya que cuenta con el ambiente necesario y una GPU de manera gratuita. Para esto, es necesario descargar el repositorio y colocarlo en un espacio en Google Drive.

El Jupyter Notebook [MainNotebook.ipynb](src/MainNotebook.ipynb) es una libreta interactiva que permite ejecutar las funciones necesarias para obtener resultados de manera no-secuencial, por lo que es posible cambiar parámetros, observar resultados y ejecutar funciones específicas.

## Instalación local
**Requisitos previos recomendados:**
- Python 3.10 (soporta también 3.9 / 3.11). 
- Git (>= 2.0)
- IDE (por ejemplo, VS Code)
- Espacio en disco para los datos y archivos preprocesados
- (Opcional) CUDA. En caso de contar con GPU. Siga los pasos indicados en https://developer.nvidia.com/cuda-downloads

### Pasos a seguir
1. Clonar el repositorio
    ```powershell
    git clone https://github.com/
    cd tu-repo/notebooks
    ```

2. Crear un ambiente virtual
    ```powershell
    # Crear ambiente virtual
    python -m venv .venv
    ```

3. Actualizar pip e instalar dependencias
    ```powershell
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    ```

4. (Opcional) Instalar PyTorch con la versión CUDA adecuada

    Si usa GPU, instale la variante de PyTorch indicada en https://pytorch.org/
    ```powershell
    # Ejemplo con CUDA 12.8
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
    ```

    Si no hay GPU, use la versión CPU:
    ```
    pip install torch torchvision
    ```

5. Abrir un IDE y ejecutar el Notebook con el ambiente creado.