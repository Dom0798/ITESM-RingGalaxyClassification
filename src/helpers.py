import cv2
import math
import torch
import shutil
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
from typing import Tuple, List
from torchvision import transforms
from astropy.io import fits
from dataclasses import dataclass
from skimage import morphology, measure
from skimage.feature import canny
from skimage.filters import unsharp_mask
from sklearn.decomposition import PCA
from skimage.measure import EllipseModel, ransac
from models import EfficientNet_2CLS, DINO_2CLS
from torch.utils.data import Dataset, DataLoader
from astropy.visualization import make_lupton_rgb, ZScaleInterval, LogStretch, MinMaxInterval


# Utils
def read_fits_file(file_path):
    with fits.open(file_path) as hdul:
        data = hdul[0].data
        r_bands = data[1]
        g_bands = data[0]
        i_bands = data[2]
    return r_bands, g_bands, i_bands


## Transfromaciones
class Transformations:
    @staticmethod
    def log_n_scale_transform(img_data, log_a=1000, clip_mode='minmax', med_val=20):
        """
        Aplicar la transformación Log-N a una imagen usando escala Zscale o Min/Max.

        Pasos del proceso:
        1. Calcular los límites de intensidades.
        2. Normalizar los valores a un rango de 0 a 1.
        3. Aplicar la transformación Log-N.
        4. Escalar la imagen de vuelta al rango original.
        5. Modificar la mediana para mantener consistencia.

        Args:
        img_data (np.ndarray): Imagen de entrada.
        log_a (float): Factor de escala para la transformación Log-N.
        clip_mode (str): Modo de escala ('zscale' o 'minmax').

        Returns:
            np.ndarray: Imagen transformada.
        """
        img = img_data.astype(np.float64)

        # 1. Calcular los límites de intensidades
        if clip_mode == 'zscale':
            interval = ZScaleInterval()
        elif clip_mode == 'minmax':
            interval = MinMaxInterval()
        p_low, p_high = interval.get_limits(img)

        # Manejar imagen plana
        if p_high <= p_low:
            return np.zeros_like(img, dtype=np.uint8)

        # 2. Normalizar los valores de pixeles entre 0-1
        x = (img - p_low) / (p_high - p_low)
        x = np.clip(x, 0.0, 1.0)

        # 3. Aplicar la transformación log
        log_stretch = LogStretch(a=log_a)
        y = log_stretch(x)

        # 4. Escalar la imagen de vuelta al rango original
        display_image = (y * 255.0).astype(np.uint8)

        # 5. Ajustar la mediana para constancia
        if med_val:
            median_val = np.median(display_image)
            rendered_img = np.clip(display_image - median_val + med_val, 0, 255).astype(np.uint8)
        else:
            rendered_img = display_image

        return rendered_img

    @staticmethod
    def rgi_stack(channel_r, channel_g, channel_i):
        """ Aplicar transformaciones a una imagen RGB.
        Args:
            channel_r (np.ndarray): Canal R.
            channel_g (np.ndarray): Canal G.
            channel_i (np.ndarray): Canal I.

        Returns:
            np.ndarray: Imagen RGB transformada.
        """
        rgb_img = make_lupton_rgb(channel_r, channel_g, channel_i, stretch=0.5, Q=8)
        return rgb_img
    
    @staticmethod
    def rgi_lognorm_stack(channel_r, channel_g, channel_i):
        """ Aplicar transformaciones LogNorm a un stack.
        Args:
            channel_r (np.ndarray): Canal R.
            channel_g (np.ndarray): Canal G.
            channel_i (np.ndarray): Canal I.

        Returns:
            np.ndarray: Imagen transformada.
        """
        rgb_img = np.stack([Transformations.log_n_scale_transform(channel_r),
                            Transformations.log_n_scale_transform(channel_g),
                            Transformations.log_n_scale_transform(channel_i)],
                            axis=-1)
        rgb_img = (np.clip(rgb_img, 0, 255)).astype(np.uint8)

        return rgb_img
    
    @staticmethod
    def rgi_unsharp_mask(channel_r, channel_g, channel_i):
        """Aplicar transformaciones Unsharp Masking a un stack.
        Args:
            channel_r (np.ndarray): Canal R.
            channel_g (np.ndarray): Canal G.
            channel_i (np.ndarray): Canal I.

        Returns:
            np.ndarray: Imagen transformada.
        """
        channel_r = Transformations.log_n_scale_transform(channel_r)
        channel_g = Transformations.log_n_scale_transform(channel_g)
        channel_i = Transformations.log_n_scale_transform(channel_i)

        r_usm = unsharp_mask(channel_r, radius=15, amount=1.5, preserve_range=False)
        g_usm = unsharp_mask(channel_g, radius=15, amount=1.5, preserve_range=False)
        i_usm = unsharp_mask(channel_i, radius=15, amount=1.5, preserve_range=False)

        rgb = np.stack([r_usm, g_usm, i_usm], axis=-1)
        rgb8 = (np.clip(rgb, 0, 255) * 255.0).astype(np.uint8)

        return rgb8
    
    @staticmethod
    def pca_stack(channel_r, channel_g, channel_i):
        """
        Aplica PCA a un stack multicanal (R, G, I) y retorna métricas clave.

        Args:
            channel_r, channel_g, channel_i (np.ndarray): Canales individuales.
            n_components (int): Número de componentes PCA.

        Returns:
            dict: {'var_ratio', 'cum_var', 'vector_resumen', 'snr_pca'}
        """
        channel_r = Transformations.log_n_scale_transform(channel_r)
        channel_g = Transformations.log_n_scale_transform(channel_g)
        channel_i = Transformations.log_n_scale_transform(channel_i)
        # Transformar cada imagen con canales G, R, I
        canales_transformados = [channel_r, channel_g, channel_i]
        # Aplanar para que quede HxWxC=# y apilar para PCA
        vectores = [canal.flatten() for canal in canales_transformados]
        matriz = np.stack(vectores, axis=0).T  # (H×W, 3)

        #n_valid asegura que no se pidan más componentes de los que se pueden calcular
        n_samples, n_features = matriz.shape
        n_valid = min(3, n_samples, n_features)

        #PCA
        pca = PCA(n_components=n_valid)
        pca_resultado = pca.fit_transform(matriz)
        
        pca_1 = pca_resultado[:, 0]
        img_pca1 = pca_1.reshape(channel_r.shape)
        img_norm = (img_pca1 - np.min(img_pca1)) / (np.max(img_pca1) - np.min(img_pca1))
        img_uint8 = (img_norm * 255).astype(np.uint8)

        return img_uint8
    
    @staticmethod
    def rgi_canny_stack(channel_r, channel_g, channel_i):
        """ Aplicar transformaciones Canny a un stack.
        Args:
            channel_r (np.ndarray): Canal R.
            channel_g (np.ndarray): Canal G.
            channel_i (np.ndarray): Canal I.

        Returns:
            np.ndarray: Imagen transformada.
        """
        # Preparar imagen en escala de grises a partir de G, R, I
        r8 = Transformations.log_n_scale_transform(channel_r).astype(np.float64)
        g8 = Transformations.log_n_scale_transform(channel_g).astype(np.float64)
        i8 = Transformations.log_n_scale_transform(channel_i).astype(np.float64)
        gray = (g8 + r8 + i8) / (3.0 * 255.0)  # normalizar a [0, 1]

        # Aplicar canny
        edges = canny(gray, sigma=1, low_threshold=0.50, high_threshold=0.90, use_quantiles=True)

        return edges


# Clasificación
class FitsInferenceDataset(torch.utils.data.Dataset):
    def __init__(self, fits_files, preprocess, transforms_dir):
        self.fits_files = fits_files
        self.preprocess = preprocess
        self.transforms_dir = transforms_dir
        self.val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.fits_files)

    def save_transformed_images(self, probs, paths, threshold, runs_dir, format='png'):
        for prob, path in zip(probs, paths):
            if format == 'png' or format == 'all':
                r_bands, g_bands, i_bands = read_fits_file(path)
                if self.preprocess == 'rgi_stack':
                    image = Transformations.rgi_stack(r_bands, g_bands, i_bands)
                elif self.preprocess == 'rgi_unsharp_mask':
                    image = Transformations.rgi_unsharp_mask(r_bands, g_bands, i_bands)

                img_name = Path(path).stem + '.png'
                if prob >= threshold:
                    dest_path = runs_dir / 'predicts' / 'rings' / 'png' / img_name
                else:
                    dest_path = runs_dir / 'predicts' / 'none' / 'png' / img_name
            
                cv2.imwrite(dest_path, image)

            if format == 'fits' or format == 'all':
                # copy original FITS file to appropriate directory
                img_name = Path(path).name
                if prob >= threshold:
                    dest_path = runs_dir / 'predicts' / 'rings' / 'fits' / img_name
                else:
                    dest_path = runs_dir / 'predicts' / 'none' / 'fits' / img_name
                shutil.copy(path, dest_path)
            
            if format not in ['png', 'fits', 'all']:
                raise ValueError("Formato no soportado. Use 'png' o 'fits'")


    def __getitem__(self, idx):
        file_path = self.fits_files[idx]
        r_bands, g_bands, i_bands = read_fits_file(file_path)

        if self.preprocess == 'rgi_stack':
            image = Transformations.rgi_stack(r_bands, g_bands, i_bands)
        elif self.preprocess == 'rgi_unsharp_mask':
            image = Transformations.rgi_unsharp_mask(r_bands, g_bands, i_bands)

        image = self.val_transform(image)

        return image, str(file_path)
    

def create_run_directory(modelo, preprocesamiento, format):
    run_id = 0
    name_folder = f'inference_{modelo}_{preprocesamiento}'
    while True:
        run_dir = Path(name_folder) / f'run_{run_id:03d}'
        predicts_dir = run_dir / 'predicts'
        ring_dir = predicts_dir / 'rings'
        none_dir = predicts_dir / 'none'
        if not run_dir.exists():
            run_dir.mkdir(parents=True)
            predicts_dir.mkdir(parents=True)
            ring_dir.mkdir(parents=True)
            none_dir.mkdir(parents=True)
            if format in ['png', 'all']:
                (ring_dir / 'png').mkdir(parents=True)
                (none_dir / 'png').mkdir(parents=True)
            if format in ['fits', 'all']:
                (ring_dir / 'fits').mkdir(parents=True)
                (none_dir / 'fits').mkdir(parents=True)
            return run_dir
        run_id += 1


def classification_inference(PREPROCESAMIENTO, MODELO, RUTA_FITS, RUTA_MODELOS, save_format='png', threshold=0.5):
    print('=== Leyendo archivos FITS de la ruta:', RUTA_FITS)
    if isinstance(RUTA_FITS, str):
        ruta_fits = Path(RUTA_FITS)
    else:
        ruta_fits = RUTA_FITS
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('CONFIG: Usando dispositivo:', device)

    # assert preprocesamiento in ['rgi_stack', 'rgi_unsharp_mask'], 'Preprocesamiento no válido'
    if PREPROCESAMIENTO in ['rgi_stack', 'rgi_unsharp_mask']:
        print('CONFIG: Usando preprocesamiento:', PREPROCESAMIENTO)
    else:
        raise ValueError('Preprocesamiento no válido', PREPROCESAMIENTO)

    # assert modelo in ['DINOv2ViTBase', 'EfficientNetB6'], 'Modelo no válido'
    if MODELO in ['DINOv2ViTBase', 'EfficientNetB6']:
        print('CONFIG: Usando modelo:', MODELO)
    else:
        raise ValueError('Modelo no válido', MODELO)


    model_path = Path(RUTA_MODELOS) / f'{MODELO}_{PREPROCESAMIENTO}.pt'
    print('=== Cargando modelo desde la ruta:', model_path)
    if MODELO == 'DINOv2ViTBase':
        model = DINO_2CLS()
        model.load_model(model_path)
    elif MODELO == 'EfficientNetB6':
        model = EfficientNet_2CLS()
        model.load_model(model_path)
    else:
        raise ValueError('Modelo no válido')

    runs_dir = create_run_directory(MODELO, PREPROCESAMIENTO, save_format)

    dataset = FitsInferenceDataset(list(ruta_fits.glob('*.fits')), PREPROCESAMIENTO, transforms_dir=runs_dir / 'transforms')
    
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)

    print('=== Iniciando inferencia...')
    probs, paths = model.predict(dataloader)
    # print(probs, paths)

    # Guardar imágenes clasificadas
    dataset.save_transformed_images(probs, paths, threshold, runs_dir, format=save_format)
    print('=== Imágenes clasificadas guardadas en:', runs_dir / 'predicts')

    # Guardar resultados en un archivo CSV
    results = []
    for prob, path in zip(probs, paths):
        results.append({'img_file': Path(path).stem, 'probability': prob, 'prediction': 'ring' if prob >= threshold else 'none'})

    results_df = pd.DataFrame(results)
    results_df.to_csv(runs_dir / 'prediction_results.csv', index=False)
    print('=== Resultados guardados en:', runs_dir / 'prediction_results.csv')
    return runs_dir


# Segmentación
def robust_scale(img, p_lo=1, p_hi=99):
    """
    Normaliza una imagen utilizando percentiles para mayor robustez ante valores extremos.
    Esto evita que píxeles muy brillantes o muy oscuros distorsionen la escala.
    """
    finite = np.isfinite(img)
    if not finite.any():
        return np.zeros_like(img, dtype=np.float32)
    # Usar percentiles en lugar de min/max para robustez
    lo, hi = np.percentile(img[finite], [p_lo, p_hi])
    if hi <= lo:
        return np.zeros_like(img, dtype=np.float32)
    # Normalizar al rango [0, 1]
    x = np.clip((img - lo) / (hi - lo), 0, 1)
    return x.astype(np.float32)


def stacked_image(r_band, g_band, i_band):
    """
    Crea una imagen apilada promediando las tres bandas GRI.
    Esto proporciona una imagen monocromática con mejor relación señal/ruido.
    """
    # Normalizar cada banda y luego promediar
    chans = [robust_scale(band) for band in (r_band, g_band, i_band)]
    return np.mean(np.stack(chans, axis=0), axis=0)


def heuristic_ring_mask(image: np.ndarray, ring_type: str) -> np.ndarray:
    """
    Genera una máscara binaria de anillo usando preprocesamiento especializado según el tipo.

    Esta es la función central del pipeline de detección. Cada tipo de anillo requiere
    una estrategia de preprocesamiento diferente para maximizar la detección:
    
    OUTER RINGS (anillos externos difusos):
    - Objetivo: Detectar estructuras débiles, difusas y alejadas del núcleo
    - Estrategia: Suavizado fuerte, supresión del núcleo, realce gamma, unsharp mask agresivo
    - Umbrales Canny bajos (0.03/0.10) para capturar bordes tenues
    - Remoción de fuentes brillantes compactas (estrellas que interfieren con RANSAC)
    
    INNER RINGS (anillos internos compactos):
    - Objetivo: Detectar estructuras compactas, bien definidas cerca del núcleo
    - Estrategia: Suavizado moderado, conectividad mejorada, anti-fragmentación
    - Umbrales Canny medios (0.05/0.15) para estructuras definidas
    - Distance transform conservador para obtener esqueleto del anillo
    
    INNER+OUTER (galaxias con dos anillos):
    - Objetivo: Detectar ambos componentes simultáneamente
    - Estrategia: Pipeline dual combinando ambos enfoques
    - Outer con estrategia difusa + Inner con estrategia compacta
    - Unión lógica de ambas máscaras
    """
    work_img = image.copy()
    ringy = None
    # ============================================================================
    # OUTER RINGS: Preprocesamiento especializado para estructuras difusas y tenues
    # ============================================================================
    if ring_type == "outer":
        # Paso 1: Suavizado fuerte para enfatizar características de gran escala
        work_img = ndi.gaussian_filter(image, sigma=3.5)
        
        # Paso 2: Suprime regiones centrales brillantes
        # Esto evita que el núcleo domine la detección
        p90 = np.percentile(work_img[work_img > 0], 90)
        bright_mask = work_img > p90
        bright_core = morphology.binary_erosion(bright_mask, morphology.disk(6))
        bright_core = morphology.binary_dilation(bright_core, morphology.disk(10))
        
        if bright_core.any():
            work_img_suppressed = work_img.copy()
            work_img_suppressed[bright_core] = np.median(work_img[~bright_core])
            work_img = work_img_suppressed
        
        # Paso 3: Realzar características tenues con ecualización adaptativa
        # Gamma < 1 amplifica valores bajos (estructuras débiles)
        work_min, work_max = work_img.min(), work_img.max()
        if work_max > work_min:
            work_norm = (work_img - work_min) / (work_max - work_min)
            work_norm = np.clip(work_norm ** 0.7, 0, 1)
            work_img = work_norm * (work_max - work_min) + work_min
        
        # Paso 4: Unsharp mask fuerte para realzar bordes difusos
        work_img = unsharp_mask(work_img, radius=4.0, amount=2.0)
        
        # Paso 5: Umbrales Canny bajos para detectar bordes tenues
        edges = canny(work_img, sigma=3.0, low_threshold=0.03, high_threshold=0.10)
        
        # Paso 6: Morfología moderada - balance entre conectividad y delgadez
        ringy = morphology.binary_dilation(edges, morphology.disk(2))
        ringy = morphology.binary_closing(ringy, morphology.disk(2))
        ringy = morphology.remove_small_objects(ringy, min_size=48)
        
        # Paso 7: Opening morfológico para suavizar protuberancias y reducir grosor
        ringy = morphology.binary_opening(ringy, morphology.disk(2))

        # Paso 8: Remueve fuentes brillantes redondas (estrellas, objetos en primer plano)
        # Estas interfieren con RANSAC al crear outliers circulares
        bright_threshold = np.percentile(image[image > 0], 99.5)
        bright_sources = image > bright_threshold
        bright_sources = morphology.binary_opening(bright_sources, morphology.disk(2))
        
        # Etiquetar y filtrar por forma: remover objetos compactos circulares
        labeled_bright = measure.label(bright_sources)
        for region in measure.regionprops(labeled_bright):
            # Mantener solo objetos muy compactos y circulares (probablemente estrellas)
            if region.area < 200 and region.eccentricity < 0.6:  # Circular y pequeño
                # Dilatar para asegurar remoción completa
                source_mask = labeled_bright == region.label
                source_mask_dilated = morphology.binary_dilation(source_mask, morphology.disk(5))
                ringy = ringy & ~source_mask_dilated
        
    # ============================================================================
    # INNER+OUTER: Pipeline dual combinando ambas estrategias
    # ============================================================================
    elif ring_type == "inner+outer":
        # COMPONENTE OUTER: usa el mismo enfoque del caso externo solo
        work_outer = ndi.gaussian_filter(image, sigma=3.5)
        p90 = np.percentile(work_outer[work_outer > 0], 90)
        bright_mask = work_outer > p90
        bright_core = morphology.binary_erosion(bright_mask, morphology.disk(6))
        bright_core = morphology.binary_dilation(bright_core, morphology.disk(10))
        
        if bright_core.any():
            work_outer[bright_core] = np.median(work_outer[~bright_core])
        
        # Realzar características tenues
        work_min, work_max = work_outer.min(), work_outer.max()
        if work_max > work_min:
            work_norm = (work_outer - work_min) / (work_max - work_min)
            work_norm = np.clip(work_norm ** 0.7, 0, 1)
            work_outer = work_norm * (work_max - work_min) + work_min
        
        # Unsharp mask fuerte
        work_outer = unsharp_mask(work_outer, radius=4.0, amount=2.0)

        # Detectar bordes externos
        outer_edges = canny(work_outer, sigma=3.0, low_threshold=0.03, high_threshold=0.10)
        ringy = morphology.binary_dilation(outer_edges, morphology.disk(2))
        ringy = morphology.binary_closing(ringy, morphology.disk(2))
        ringy = morphology.remove_small_objects(ringy, min_size=48)
        ringy = morphology.binary_opening(ringy, morphology.disk(2))
        
        # Remover fuentes brillantes redondas 
        bright_threshold = np.percentile(image[image > 0], 99.5)
        bright_sources = image > bright_threshold
        bright_sources = morphology.binary_opening(bright_sources, morphology.disk(2))
        
        labeled_bright = measure.label(bright_sources)
        for region in measure.regionprops(labeled_bright):
            if region.area < 200 and region.eccentricity < 0.6:
                source_mask = labeled_bright == region.label
                source_mask_dilated = morphology.binary_dilation(source_mask, morphology.disk(5))
                ringy = ringy & ~source_mask_dilated
        
        # COMPONENTE INNER: preprocesamiento diferente para características compactas
        inner_work = ndi.gaussian_filter(image, sigma=1.6)
        inner_edges = canny(inner_work, sigma=1.8, low_threshold=0.05, high_threshold=0.15)
        inner = morphology.binary_dilation(inner_edges, morphology.disk(2))
        inner = ndi.binary_fill_holes(inner)
        inner = morphology.remove_small_objects(inner, min_size=48)
        
        # Combinar ambos componentes
        ringy = np.logical_or(ringy, inner)
        ringy = morphology.binary_closing(ringy, morphology.disk(2))
        
    # ============================================================================
    # INNER RINGS: Preprocesamiento para estructuras compactas y bien definidas
    # ============================================================================
    elif ring_type == "inner":
        # Suavizado moderado para inner rings
        work_img = ndi.gaussian_filter(image, sigma=1.6)
        edges = canny(work_img, sigma=1.8, low_threshold=0.05, high_threshold=0.15)

        # Construir máscara conectada con closing antes de rellenar huecos
        ringy = morphology.binary_dilation(edges, morphology.disk(2))
        ringy = morphology.binary_closing(ringy, morphology.disk(3))  
        ringy = ndi.binary_fill_holes(ringy)
        
        # Remover fragmentos pequeños
        ringy = morphology.remove_small_objects(ringy, min_size=100)  # Incrementado de 64
        
        # Refinamiento suave para obtener estructura de anillo sin fragmentar
        ringy = morphology.binary_erosion(ringy, morphology.disk(1))  
        ringy = morphology.binary_dilation(ringy, morphology.disk(2))

        # Usar distance transform conservadoramente
        # Esto obtiene el "esqueleto" del anillo sin adelgazarlo demasiado
        dist = ndi.distance_transform_edt(ringy)
        core = dist >= 1.5  
        if core.any() and np.sum(core) > 50:  # Solo usar si el resultado es sustancial
            ringy = core
            # Rellenar pequeños gaps creados por distance transform
            ringy = morphology.binary_closing(ringy, morphology.disk(2))
        
        # Limpieza final
        ringy = morphology.remove_small_objects(ringy, min_size=48)
            
    return ringy.astype(np.uint8)


def extract_ring_masks(mask: np.ndarray,
                       ring_type: str,
                       image=None):
    """
    Separa componentes individuales de una máscara binaria y los clasifica (outer, inner).
    
    Esta función es crítica para detectar correctamente anillos múltiples (inner+outer).
    Usa un sistema de puntuación inteligente para priorizar componentes periféricos y difusos
    sobre componentes centrales y brillantes cuando se buscan anillos externos.

    Sistema de puntuación para anillos externos:
    - Área (50%): Componentes grandes son preferidos (anillos genuinos vs fragmentos)
    - Periferalidad (30%): Componentes alejados del centro son preferidos (externos vs internos)
    - Penalización por brillo (20%): Componentes tenues son preferidos (anillos difusos vs núcleos brillantes)

    Estrategia dual para detección de anillo interno en casos inner+outer:
    1. Primer intento: Buscar con heurística específica para anillos internos (más confiable)
    2. Segundo intento: Buscar en componentes restantes con filtros geométricos estrictos
    
    Args:
        mask: Máscara binaria con posibles anillos detectados
        ring_type: Tipo de anillo esperado ('outer', 'inner', 'inner+outer')
        image: Imagen original (necesaria para scoring inteligente)
    
    Returns:
        Lista de tuplas (etiqueta, máscara) para cada componente detectado
    """
    mask_bool = mask.astype(bool)
    if not mask_bool.any():
        return []
    
    # Etiquetar componentes conectados
    labeled = measure.label(mask_bool, connectivity=2)
    regions = [r for r in measure.regionprops(labeled) if r.area >= 48]
    if not regions:
        return []
    
    # ANILLOS EXTERNOS: Priorizar componentes periféricos y difusos sobre centrales y brillantes
    # Esto es crucial porque los anillos externos tienden a ser más débiles y alejados del núcleo
    if ring_type in ("outer", "inner+outer") and image is not None:
        # Calcular centro de la imagen como referencia
        img_center_y, img_center_x = np.array(image.shape) / 2.0
        
        # Sistema de puntuación: área (50%) + periferalidad (30%) - brillo (20%)
        # Componentes grandes, periféricos y tenues obtienen mayor puntuación
        scored_regions = []
        for r in regions:
            cy, cx = r.centroid
            dist_from_center = np.sqrt((cx - img_center_x)**2 + (cy - img_center_y)**2)
            
            # Extraer intensidad promedio en esta región
            # Para outer: menor intensidad es mejor (anillos difusos vs núcleos brillantes)
            region_mask = labeled == r.label
            mean_intensity = np.mean(image[region_mask])
            
            # Normalizar componentes del score para comparación justa
            peripherality_score = dist_from_center / max(image.shape)  # [0, ~1]
            area_score = r.area / max([rr.area for rr in regions])      # [0, 1]
            intensity_penalty = mean_intensity / (np.mean(image) + 1e-6)  # Relativo al promedio
            
            # Combinar: favorece grande + periférico + tenue
            score = area_score * 0.5 + peripherality_score * 0.3 - intensity_penalty * 0.2
            scored_regions.append((score, r))
        
        # Ordenar por puntuación: mejor candidato primero
        scored_regions.sort(key=lambda x: x[0], reverse=True)
        regions = [r for _, r in scored_regions]
    else:
        # Para inner o casos sin imagen: simplemente ordenar por área (más simple)
        regions.sort(key=lambda r: r.area, reverse=True)

    components = []

    # CASO ESPECIAL: INNER+OUTER (galaxias con dos anillos distintos)
    # Requiere estrategia dual para separar correctamente ambos componentes
    if ring_type == "inner+outer":
        # Paso 1: Identificar el anillo externo (ya ordenado por scoring, el primero es el mejor)
        outer_region = regions[0]
        outer_mask = (labeled == outer_region.label).astype(np.uint8)
        components.append(("outer", outer_mask))

        # Extraer propiedades geométricas del externo para validación del interno
        outer_cy, outer_cx = outer_region.centroid
        outer_major = max(outer_region.major_axis_length, 1e-3)
        outer_minor = max(outer_region.minor_axis_length, 1e-3)
        outer_area = outer_region.area

        # Definir región interior donde debe estar el anillo interno
        # Rellenamos huecos y erosionamos para obtener el área contenida por el externo
        outer_interior = ndi.binary_fill_holes(outer_mask.astype(bool))
        outer_interior = morphology.binary_erosion(outer_interior, morphology.disk(3))
        if not outer_interior.any():
            outer_interior = outer_mask.astype(bool)

        # Paso 2a: ESTRATEGIA PRIMARIA - Buscar anillo interno con heurística específica
        # Esto es más confiable que buscar en componentes ya detectados
        inner_mask = None
        if image is not None:
            # Aplicar detección especializada para anillos internos
            inner_candidate = heuristic_ring_mask(image, "inner").astype(bool)
            # Restringir búsqueda al interior del externo
            inner_candidate &= outer_interior
            inner_candidate = morphology.remove_small_objects(inner_candidate, min_size=64)
            
            if inner_candidate.any():
                labeled_inner = measure.label(inner_candidate, connectivity=2)
                inner_regions = [r for r in measure.regionprops(labeled_inner) if r.area >= 64]
                
                # Filtros de validación geométrica (relajados para mayor sensibilidad)
                valid_inner = []
                for r in inner_regions:
                    # Debe ser más pequeño que el externo (15% del tamaño)
                    if r.major_axis_length < outer_major * 0.15:
                        continue
                    # Debe tener área significativa (3% del externo)
                    if r.area < outer_area * 0.03:
                        continue
                    # Debe ser elongado, no circular (aspecto < 0.9)
                    aspect_ratio = r.minor_axis_length / max(r.major_axis_length, 1e-3)
                    if aspect_ratio > 0.9:
                        continue
                    # Debe tener excentricidad mínima (Evita nucleos galacticos y estrellas)
                    if r.eccentricity < 0.3:
                        continue
                    # Debe ser razonablemente compacto (solidity > 0.6)
                    if r.solidity < 0.6:
                        continue
                    valid_inner.append(r)
                
                if valid_inner:
                    # Preferir candidatos grandes y elongados
                    valid_inner.sort(key=lambda r: r.area * (1 - aspect_ratio), reverse=True)
                    inner_mask = (labeled_inner == valid_inner[0].label).astype(np.uint8)

        # Paso 2b: ESTRATEGIA SECUNDARIA - Buscar en componentes restantes
        # Solo si la estrategia primaria falló
        if inner_mask is None:
            for region in regions[1:]:  # Saltar el anillo externo (ya procesado)
                # Debe ser estrictamente menor que el externo
                if region.major_axis_length >= outer_major or region.minor_axis_length >= outer_minor:
                    continue
                # Filtros de tamaño mínimo
                if region.major_axis_length < outer_major * 0.15:
                    continue
                if region.area < outer_area * 0.03:
                    continue
                
                # Filtros de forma: debe ser elongado
                aspect_ratio = region.minor_axis_length / max(region.major_axis_length, 1e-3)
                if aspect_ratio > 0.9:
                    continue
                # Excentricidad mínima (Evita núcleos galácticos y estrellas)
                if region.eccentricity < 0.25:
                    continue
                # Validación adicional: rechazar estructuras muy sólidas (barras centrales)
                if region.solidity < 0.6:
                    continue

                # Verificar que esté contenido dentro del anillo externo (coordenadas elípticas)
                cy, cx = region.centroid
                norm_x = (cx - outer_cx) / (outer_major * 0.5)
                norm_y = (cy - outer_cy) / (outer_minor * 0.5)
                # Debe estar dentro de la elipse del anillo externo (con margen 5%)
                if norm_x**2 + norm_y**2 <= 1.05:
                    candidate = ((labeled == region.label).astype(bool)) & outer_interior
                    candidate = morphology.remove_small_objects(candidate, min_size=64)
                    if candidate.any():
                        inner_mask = candidate.astype(np.uint8)
                        break

        # Agregar el anillo interno si se encontró
        if inner_mask is not None:
            components.append(("inner", inner_mask))
    
    # CASOS SIMPLES: OUTER, INNER o NONE
    # Solo hay un componente dominante
    else:
        dominant = regions[0]
        label_name = ring_type if ring_type not in ("none", "") else "disk"
        components = [(label_name, (labeled == dominant.label).astype(np.uint8))]

    return components


def _ellipse_from_mask(mask: np.ndarray):
    """
    Ajusta una elipse a una máscara binaria usando RANSAC para robustez ante ruido.
    Este es el método principal para convertir una máscara detectada en parámetros geométricos.
    
    Pasos:
    1. Limpia la máscara de fragmentos pequeños (estrellas, ruido)
    2. Extrae el contorno principal
    3. Ajusta elipse con RANSAC (robusto ante outliers)
    4. Valida que la elipse sea elongada (no circular) y de tamaño razonable
    """
    # Preprocesar: eliminar componentes aislados pequeños (estrellas/ruido)
    mask_cleaned = morphology.remove_small_objects(mask.astype(bool), min_size=64)
    mask_cleaned = morphology.binary_closing(mask_cleaned, morphology.disk(3))
    
    if not mask_cleaned.any():
        return None
    
    # Extraer contornos de la máscara limpia
    contours = measure.find_contours(mask_cleaned, level=0.5)
    if not contours:
        return None
    
    # Usar el contorno más largo (estructura principal del anillo)
    contours_sorted = sorted(contours, key=len, reverse=True)
    main_contour = contours_sorted[0]
    
    # Validar que el contorno tenga suficientes puntos
    if len(main_contour) < 30:
        return None
    
    # Convertir a coordenadas xy (scikit-image devuelve yx)
    pts_xy = np.fliplr(main_contour)
    
    # Submuestrear si hay demasiados puntos (acelera RANSAC)
    if len(pts_xy) > 500:
        indices = np.linspace(0, len(pts_xy) - 1, 500, dtype=int)
        pts_xy = pts_xy[indices]
    
    model = EllipseModel()
    try:
        # Usar RANSAC con tolerancia relajada para aceptar anillos difusos/pequeños
        model_robust, inliers = ransac(
            pts_xy, 
            EllipseModel, 
            min_samples=10,           # Mínimo de puntos para ajustar
            residual_threshold=4.0,   # Tolerancia en píxeles
            max_trials=5000           # Intentos máximos
        )
        
        if model_robust is None:
            return None
        
        # Extraer parámetros para validación
        xc, yc, a, b, theta = model_robust.params
        if b > a:
            a, b = b, a
        
        # Validar aspecto: rechazar ajustes casi circulares (pueden ser discos, no anillos)
        aspect_ratio = b / max(a, 1e-6)
        if aspect_ratio > 0.9:
            return None
        
        # Validar tamaño mínimo
        if a < 10 or b < 7:
            return None
        
        # Validar calidad del ajuste: ratio de inliers debe ser razonable
        if inliers is not None:
            inlier_ratio = np.sum(inliers) / len(inliers)
            if inlier_ratio < 0.25:  # Al menos 25% de puntos deben seguir la elipse
                return None
        
        return model_robust
    except Exception:
        return None


def _ellipse_to_params(model: EllipseModel):
    """
    Extrae y normaliza los parámetros de una elipse.
    Asegura que 'a' sea siempre el semieje mayor.
    """
    xc, yc, a, b, theta = model.params
    # Asegurar que a >= b (convención: a es semieje mayor)
    if b > a:
        a, b = b, a
        theta = (theta + np.pi / 2) % np.pi
    return float(xc), float(yc), float(a), float(b), float(theta)


def _radial_width(mask, xc, yc, a, b, theta):
    """
    Estima el ancho radial del anillo usando la distribución de distancias elípticas.
    Usa cuartiles (Q75 - Q25) para robustez ante valores extremos.
    """
    ys, xs = np.nonzero(mask)
    if len(xs) < 20:
        return math.nan
    
    # Transformar al sistema de coordenadas de la elipse
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    x = xs - xc
    y = ys - yc
    xp = x * cos_t + y * sin_t
    yp = -x * sin_t + y * cos_t
    
    # Calcular distancia elíptica para cada píxel
    q = max(b, 1e-6) / max(a, 1e-6)
    r_e = np.sqrt(xp**2 + (yp / q) ** 2)
    
    # Usar rango intercuartil como medida robusta del ancho
    q25, q75 = np.percentile(r_e, [25, 75])
    return float(q75 - q25)


def _surface_brightness_contrast(image, xc, yc, a, b, theta, dr_pix=3.0):
    """
    Calcula el contraste de brillo superficial entre el anillo y el gap interior.
    Un valor positivo indica que el anillo es más brillante que el gap (anillo en emisión).
    Retorna la diferencia en magnitudes.
    """
    # Crear grilla de coordenadas
    yy, xx = np.indices(image.shape)
    
    # Transformar a coordenadas elípticas
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    x = xx - xc
    y = yy - yc
    xp = x * cos_t + y * sin_t
    yp = -x * sin_t + y * cos_t
    q = max(b, 1e-6) / max(a, 1e-6)
    r_e = np.sqrt(xp**2 + (yp / q) ** 2)
    
    # Definir regiones: anillo y gap interior
    ring_mask = (r_e >= (a - dr_pix)) & (r_e <= (a + dr_pix))
    gap_mask = (r_e >= (a - 2 * dr_pix)) & (r_e < (a - dr_pix))
    
    # Extraer intensidades
    I_ring = image[ring_mask]
    I_gap = image[gap_mask]
    I_ring = I_ring[np.isfinite(I_ring)]
    I_gap = I_gap[np.isfinite(I_gap)]
    
    if len(I_ring) < 50 or len(I_gap) < 50:
        return math.nan
    
    # Convertir a magnitudes (brillo superficial)
    eps = 1e-6
    mu_ring = -2.5 * np.log10(np.clip(I_ring.mean(), eps, None))
    mu_gap = -2.5 * np.log10(np.clip(I_gap.mean(), eps, None))
    
    # Contraste: negativo si anillo es más brillante (menor magnitud)
    return float(mu_ring - mu_gap)


@dataclass
class RingMeasurement:
    """
    Almacena todas las propiedades geométricas y fotométricas de un anillo detectado.
    Incluye parámetros de la elipse ajustada, dimensiones físicas y contraste.
    """
    ring_label: str              # Tipo de anillo: 'inner', 'outer', etc.
    center_xy: Tuple[float, float]  # Centro (x, y) en píxeles
    a_pix: float                 # Semieje mayor en píxeles
    b_pix: float                 # Semieje menor en píxeles
    pa_deg: float                # Ángulo de posición en grados
    width_pix: float             # Ancho radial del anillo en píxeles
    width_as: float              # Ancho radial en arcosegundos
    a_as: float                  # Semieje mayor en arcosegundos
    b_as: float                  # Semieje menor en arcosegundos
    circumference_pix: float     # Circunferencia en píxeles
    circumference_as: float      # Circunferencia en arcosegundos
    contrast_mag: float          # Contraste en magnitudes
    method: str                  # Método de ajuste:


def ellipse_circumference(a: float, b: float) -> float:
    """
    Calcula la circunferencia aproximada de una elipse usando la fórmula de Ramanujan.
    Esta aproximación es muy precisa para todo tipo de elipticidades.
    """
    if a <= 0 or b <= 0:
        return math.nan
    # Fórmula de Ramanujan para circunferencia de elipse
    h = ((a - b) ** 2) / ((a + b) ** 2)
    return math.pi * (a + b) * (1 + (3 * h) / (10 + math.sqrt(max(0.0, 4 - 3 * h))))


def measure_ring_from_mask(image,
                           mask,
                           pixscale_as=0.262,
                           ring_label: str = "disk"):
    """
    Ajusta una elipse a una máscara de anillo usando RANSAC y calcula sus propiedades.
    
    Pipeline de medición en 4 pasos:
    1. Limpieza de máscara: remover estrellas y artefactos pequeños
    2. Ajuste RANSAC: método robusto para elipses elongadas
    3. Validación: verificar solapamiento con máscara y posición del centro
    4. Cálculo de propiedades: ancho radial, contraste, circunferencia
    
    Criterios de rechazo:
    - Aspect ratio > 0.9 (demasiado circular, probablemente no un anillo)
    - Tamaño mínimo: a < 8px o b < 6px
    - Solapamiento con máscara < 20%
    - Centro desplazado > 60% del semieje mayor
    """
    # Paso 1: Limpiar máscara de estrellas y fragmentos pequeños
    mask = morphology.remove_small_objects(mask.astype(bool), min_size=64)
    mask = morphology.binary_opening(mask, morphology.disk(2))
    mask = morphology.binary_closing(mask, morphology.disk(4))
    mask = mask.astype(np.uint8)
    if mask.sum() == 0:
        return None
    
    # Paso 2: Ajuste RANSAC (único método, sin fallback)
    model = _ellipse_from_mask(mask)
    if model is None:
        return None
    
    # Paso 3: Validar resultado de RANSAC
    xc, yc, a, b, theta = _ellipse_to_params(model)
    
    # Validación 1: Rechazar ajustes casi circulares
    aspect_ratio = b / max(a, 1e-6)
    if aspect_ratio > 0.9:
        return None
    
    # Validación 2: Rechazar tamaños muy pequeños
    if a < 8 or b < 6:
        return None
    
    # Validación 3: Verificar que la elipse tenga sentido dado la máscara
    yy, xx = np.indices(mask.shape)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    x = xx - xc
    y = yy - yc
    xp = x * cos_t + y * sin_t
    yp = -x * sin_t + y * cos_t
    q = max(b, 1e-6) / max(a, 1e-6)
    r_e = np.sqrt(xp**2 + (yp / q) ** 2)
    
    # Verificar solapamiento entre elipse ajustada y máscara real
    ellipse_ring = (r_e >= a * 0.75) & (r_e <= a * 1.25)
    overlap = np.sum(ellipse_ring & mask.astype(bool)) / max(np.sum(ellipse_ring), 1)
    
    # Verificar que la elipse no esté centrada en un punto brillante lejos de la máscara
    mask_centroid_y, mask_centroid_x = ndi.center_of_mass(mask)
    dist_from_mask_center = np.sqrt((xc - mask_centroid_x)**2 + (yc - mask_centroid_y)**2)
    max_allowed_offset = max(a, b) * 0.6
    
    # Rechazar si solapamiento bajo o centro muy desplazado
    if overlap < 0.2 or dist_from_mask_center > max_allowed_offset:
        return None
    
    # Paso 4: Calcular propiedades geométricas y fotométricas
    width_pix = _radial_width(mask, xc, yc, a, b, theta)
    contrast = _surface_brightness_contrast(image, xc, yc, a, b, theta, dr_pix=max(3, 0.05 * a))
    circ_pix = ellipse_circumference(a, b)
    circ_as = circ_pix * pixscale_as if math.isfinite(circ_pix) else math.nan
    
    return RingMeasurement(
        ring_label=ring_label,
        center_xy=(xc, yc),
        a_pix=a,
        b_pix=b,
        pa_deg=theta * 180.0 / math.pi,
        width_pix=width_pix,
        width_as=width_pix * pixscale_as if math.isfinite(width_pix) else math.nan,
        a_as=a * pixscale_as,
        b_as=b * pixscale_as,
        circumference_pix=circ_pix,
        circumference_as=circ_as,
        contrast_mag=contrast,
        method="ransac",
    )


def segment_and_measure_ring(fits_path: Path,
                             ring_type: str,
                             pixscale_as: float = 0.262):
    """
    Pipeline completo de segmentación y medición de anillos galácticos.
    
    Flujo en 5 pasos:
    1. Leer archivo FITS y extraer bandas GRI
    2. Apilar bandas para obtener imagen monocromática con mejor SNR
    3. Generar máscara heurística especializada según tipo de anillo
    4. Extraer y separar componentes individuales (outer/inner)
    5. Ajustar elipses con RANSAC y calcular propiedades para cada componente
    
    Returns:
        tuple: (imagen apilada, máscara binaria, lista de mediciones)
    """
    r_band, g_band, i_band = read_fits_file(fits_path)
    img = stacked_image(r_band, g_band, i_band)
    mask = heuristic_ring_mask(img, ring_type)
    ring_masks = extract_ring_masks(mask, ring_type, image=img)
    measurements = []
    for label_name, component_mask in ring_masks:
        meas = measure_ring_from_mask(img, component_mask,
                                      pixscale_as=pixscale_as,
                                      ring_label=label_name)
        if meas is not None:
            measurements.append(meas)
    return img, mask, measurements


_RING_COLORS = {
    "outer": "#ff7f0e",
    "inner": "#1f77b4"
}


def ring_segmentation_inference(fits_path, ring_type, show_mask, runs_dir, display=True):
    if isinstance(runs_dir, str):
        runs_dir = Path(runs_dir)

    if isinstance(fits_path, str):
        fits_path = Path(fits_path)
    
    segmentation_dir = runs_dir / 'segmentations'

    if not segmentation_dir.exists():
        segmentation_dir.mkdir(parents=True)
    print(f'CONFIG: Usando archivo {fits_path.name}')
    print(f'CONFIG: Buscando anillos de tipo {ring_type}')
    print('=== Generando segmentación y mediciones...')
    img, mask, measurements = segment_and_measure_ring(fits_path, ring_type)
    if not measurements:
        print(f'No se encontraron anillos de tipo {ring_type} en {fits_path.name}.')
        return
    canvas = img.copy()
    if canvas.ndim == 2:  # grayscale
        canvas = (np.clip(canvas, 0, 1) * 255).astype(np.uint8)
        canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)

    if show_mask:
        mask_bool = mask.astype(bool)
        overlay = canvas.copy()
        overlay[:] = (255, 0, 0)  # Red color for the mask
        alpha = 0.4  # Transparency factor
        canvas[mask_bool] = cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0)[mask_bool]

    for meas_idx, meas in enumerate(measurements):
        color = _RING_COLORS.get(meas.ring_label, _RING_COLORS.get(ring_type, "#2ca02c"))
        print(f'Anillo {meas.ring_label} encontrado!')

        center = tuple(map(int, meas.center_xy))
        axes = (int(meas.a_pix), int(meas.b_pix))
        angle = meas.pa_deg
        rgb_color = tuple(int(color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
        cv2.ellipse(canvas, center, axes, angle, 0, 360, rgb_color, 1)

        ax1_p1 = (int(round(meas.center_xy[0] - meas.a_pix * math.cos(math.radians(angle)))),
                  int(round(meas.center_xy[1] - meas.a_pix * math.sin(math.radians(angle)))))
        ax1_p2 = (int(round(meas.center_xy[0] + meas.a_pix * math.cos(math.radians(angle)))),
                  int(round(meas.center_xy[1] + meas.a_pix * math.sin(math.radians(angle)))))
        ax2_p1 = (int(round(meas.center_xy[0] - meas.b_pix * math.sin(math.radians(angle)))),
                  int(round(meas.center_xy[1] + meas.b_pix * math.cos(math.radians(angle)))))
        ax2_p2 = (int(round(meas.center_xy[0] + meas.b_pix * math.sin(math.radians(angle)))),
                  int(round(meas.center_xy[1] - meas.b_pix * math.cos(math.radians(angle)))))

        cv2.line(canvas, ax1_p1, ax1_p2, rgb_color, thickness=1, lineType=cv2.LINE_AA)
        cv2.line(canvas, ax2_p1, ax2_p2, rgb_color, thickness=1, lineType=cv2.LINE_AA)

        cv2.putText(canvas,
                    f"{meas.ring_label.capitalize()} C={meas.circumference_pix:.1f}px",
                    (0, 10 * (meas_idx  + 1)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.3,
                    rgb_color,
                    thickness=1,
                    lineType=cv2.LINE_AA)

        if not (runs_dir / 'segmentation_results.csv').exists():
            with open(runs_dir / 'segmentation_results.csv', 'w') as f:
                f.write("fits_name,scale,ring_label,center_x_pix,center_y_pix,a_pix,b_pix,pa_deg,circumference_pix\n")
                f.write(f"{fits_path.name},0.262,{meas.ring_label},{meas.center_xy[0]},{meas.center_xy[1]},{meas.a_pix},{meas.b_pix},{meas.pa_deg},{meas.circumference_pix}\n")
        else:
            with open(runs_dir / 'segmentation_results.csv', 'a') as f:
                f.write(f"{fits_path.name},0.262,{meas.ring_label},{meas.center_xy[0]},{meas.center_xy[1]},{meas.a_pix},{meas.b_pix},{meas.pa_deg},{meas.circumference_pix}\n")

        # delete duplicates in csv
        df = pd.read_csv(runs_dir / 'segmentation_results.csv')
        df = df.drop_duplicates()
        df.to_csv(runs_dir / 'segmentation_results.csv', index=False)
        print('=== Mediciones guardadas en:', runs_dir / 'segmentation_results.csv')

    if display:
        plt.axis('off')
        plt.imshow(canvas)
        
    cv2.imwrite(str(segmentation_dir / f"{fits_path.stem}.png"), cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    print('=== Segmentación guardada en:', segmentation_dir)