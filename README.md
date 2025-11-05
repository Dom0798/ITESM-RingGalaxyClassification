# ITESM-RingGalaxyClassification
Este repositorio contiene un proyecto realizado por alumnes del ITESM de la maestría de Inteligencia Artificial en colaboración con el Instituto de Astronomía de la UNAM.

El proyecto se divide en dos secciones:
- Clasificación de galaxias (anillo / no anillo)
- Segmentación de anillos según su tipo (interno, externo o interno+externo)

## Datos a utilizar
Los datos utilizados en este proyecto corresponen a [DESI Legacy Survey](https://www.legacysurvey.org/) en la versión DR10.
Los archivos FITS deben tener la siguiente configuración:
- Escala: 0.262
- Tamaño de imagen: 224x224
- Bandas: gri

## Inicio rápido
La manera más rápida de inferir y obtener segmentaciones es utilizar Google Colaboratory utilizando el Jupyter Notebook [MainNotebook.ipynb](src/MainNotebook.ipynb)
