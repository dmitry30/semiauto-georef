import cv2
import numpy as np
from osgeo import gdal, osr
import tempfile
import os
import uuid
from pathlib import Path

# Включаем исключения GDAL для лучшей отладки
gdal.UseExceptions()

class Georeferencer:
    """Простой геопривязчик на основе особых точек"""
    
    def __init__(self, temp_dir="temp"):
        # Используем ORB - бесплатный и работает без лицензионных проблем
        self.detector = cv2.ORB_create(nfeatures=5000)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(exist_ok=True)
    
    def match_images(self, img1, img2):
        """Найти соответствия между двумя изображениями"""
        # Конвертируем в grayscale
        if len(img1.shape) == 3:
            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        else:
            gray1 = img1
            
        if len(img2.shape) == 3:
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        else:
            gray2 = img2
        
        # Находим ключевые точки и дескрипторы
        kp1, des1 = self.detector.detectAndCompute(gray1, None)
        kp2, des2 = self.detector.detectAndCompute(gray2, None)
        
        print(f"Found {len(kp1) if kp1 else 0} keypoints in source")
        print(f"Found {len(kp2) if kp2 else 0} keypoints in reference")
        
        if des1 is None or des2 is None or len(des1) < 10 or len(des2) < 10:
            return None, "Недостаточно ключевых точек для сопоставления"
        
        # Ищем соответствия
        matches = self.matcher.knnMatch(des1, des2, k=2)
        
        # Фильтруем по расстоянию Лоу
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)
        
        print(f"Found {len(good_matches)} good matches")
        
        if len(good_matches) < 10:
            return None, f"Недостаточно хороших соответствий: {len(good_matches)}"
        
        # Подготавливаем точки для гомографии
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1,1,2)
        
        # Находим гомографию с RANSAC
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        if H is None:
            return None, "Не удалось найти преобразование"
        
        # Фильтруем matches по маске RANSAC
        matches_mask = mask.ravel().tolist()
        inlier_matches = [m for m, mask_val in zip(good_matches, matches_mask) if mask_val]
        
        print(f"Found {len(inlier_matches)} inlier matches after RANSAC")
        
        return {
            'homography': H,
            'matches': inlier_matches,
            'keypoints1': kp1,
            'keypoints2': kp2,
            'num_inliers': len(inlier_matches)
        }, None
    
    def create_geotiff(self, image_array, bounds, output_path):
        """
        Создать GeoTIFF файл из массива пикселей с заданными границами
        bounds: {'top_left': (lat, lon), 'bottom_right': (lat, lon)}
        """
        height, width = image_array.shape[:2]
        
        # Создаем dataset
        driver = gdal.GetDriverByName('GTiff')
        
        # Определяем количество каналов
        if len(image_array.shape) == 3 and image_array.shape[2] == 3:
            num_bands = 3
            data_type = gdal.GDT_Byte
        else:
            num_bands = 1
            data_type = gdal.GDT_Byte
        
        print(f"Creating GeoTIFF: {width}x{height}, {num_bands} bands, path: {output_path}")
        
        dataset = driver.Create(
            str(output_path), 
            width, 
            height, 
            num_bands,
            data_type,
            options=['COMPRESS=LZW']  # сжатие для уменьшения размера
        )
        
        if dataset is None:
            raise Exception(f"Не удалось создать GeoTIFF файл: {output_path}")
        
        # Устанавливаем геотрансформ
        top_left_lat, top_left_lon = bounds['top_left']
        bottom_right_lat, bottom_right_lon = bounds['bottom_right']
        
        pixel_width = (bottom_right_lon - top_left_lon) / width
        pixel_height = (top_left_lat - bottom_right_lat) / height
        
        geotransform = (
            top_left_lon,    # верхний левый X (долгота)
            pixel_width,     # разрешение по X (градусы на пиксель)
            0,               # поворот (обычно 0)
            top_left_lat,    # верхний левый Y (широта)
            0,               # поворот (обычно 0)
            -pixel_height    # разрешение по Y (отрицательное, так как Y идет вниз)
        )
        
        dataset.SetGeoTransform(geotransform)
        
        # Устанавливаем проекцию (WGS84)
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(4326)  # WGS84
        dataset.SetProjection(srs.ExportToWkt())
        
        # Записываем данные
        if num_bands == 3:
            # OpenCV хранит в BGR, GDAL ожидает RGB
            # Записываем каналы в правильном порядке
            band_r = dataset.GetRasterBand(1)
            band_g = dataset.GetRasterBand(2)
            band_b = dataset.GetRasterBand(3)
            
            # BGR -> RGB
            band_r.WriteArray(image_array[:, :, 2])  # Red channel
            band_g.WriteArray(image_array[:, :, 1])  # Green channel
            band_b.WriteArray(image_array[:, :, 0])  # Blue channel
        else:
            # Grayscale
            band = dataset.GetRasterBand(1)
            band.WriteArray(image_array)
        
        # Устанавливаем метаданные
        for i in range(num_bands):
            dataset.GetRasterBand(i + 1).SetNoDataValue(0)
        
        # Закрываем dataset
        dataset.FlushCache()
        dataset = None
        
        print(f"GeoTIFF создан: {output_path}")
        return output_path
    
    def apply_georeference(self, source_image, reference_image, reference_bounds):
        """
        Основной метод: привязать исходное изображение к референсу
        """
        # Конвертируем PIL в numpy для OpenCV
        source_np = np.array(source_image)
        reference_np = np.array(reference_image)
        
        print(f"Source shape: {source_np.shape}, Reference shape: {reference_np.shape}")
        
        # PIL -> OpenCV (RGB -> BGR)
        if len(source_np.shape) == 3 and source_np.shape[2] == 3:
            source_np = cv2.cvtColor(source_np, cv2.COLOR_RGB2BGR)
        if len(reference_np.shape) == 3 and reference_np.shape[2] == 3:
            reference_np = cv2.cvtColor(reference_np, cv2.COLOR_RGB2BGR)
        
        # Ищем соответствия
        print("Matching images...")
        result, error = self.match_images(source_np, reference_np)
        if error:
            return None, error
        
        print(f"Homography matrix:\n{result['homography']}")
        
        # Применяем гомографию к координатам углов исходного изображения
        h, w = source_np.shape[:2]
        corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        
        transformed_corners = cv2.perspectiveTransform(corners, result['homography'])
        
        print(f"Original corners: {corners.reshape(-1, 2)}")
        print(f"Transformed corners: {transformed_corners.reshape(-1, 2)}")
        
        # Находим новые границы в пикселях референса
        min_x = transformed_corners[:, 0, 0].min()
        max_x = transformed_corners[:, 0, 0].max()
        min_y = transformed_corners[:, 0, 1].min()
        max_y = transformed_corners[:, 0, 1].max()
        
        print(f"Bounds in reference pixels: x=[{min_x:.1f}, {max_x:.1f}], y=[{min_y:.1f}, {max_y:.1f}]")
        
        # Преобразуем пиксельные координаты в географические
        ref_h, ref_w = reference_np.shape[:2]
        top_left_lat, top_left_lon = reference_bounds['top_left']
        bottom_right_lat, bottom_right_lon = reference_bounds['bottom_right']
        
        print(f"Reference bounds: TL=({top_left_lat:.6f}, {top_left_lon:.6f}), BR=({bottom_right_lat:.6f}, {bottom_right_lon:.6f})")
        
        pix_per_lon = (bottom_right_lon - top_left_lon) / ref_w
        pix_per_lat = (top_left_lat - bottom_right_lat) / ref_h
        
        # Новые границы для исходного изображения
        new_top_left_lon = top_left_lon + (min_x * pix_per_lon)
        new_top_left_lat = top_left_lat - (min_y * pix_per_lat)
        new_bottom_right_lon = top_left_lon + (max_x * pix_per_lon)
        new_bottom_right_lat = top_left_lat - (max_y * pix_per_lat)
        
        print(f"New bounds: TL=({new_top_left_lat:.6f}, {new_top_left_lon:.6f}), BR=({new_bottom_right_lat:.6f}, {new_bottom_right_lon:.6f})")
        
        new_bounds = {
            'top_left': (new_top_left_lat, new_top_left_lon),
            'bottom_right': (new_bottom_right_lat, new_bottom_right_lon)
        }
        
        # Создаем уникальное имя для выходного файла
        output_filename = f"georef_{uuid.uuid4().hex[:8]}.tif"
        output_path = self.temp_dir / output_filename
        
        # Создаем GeoTIFF
        try:
            self.create_geotiff(source_np, new_bounds, output_path)
            return str(output_path), f"Успешно найдено {result['num_inliers']} соответствий"
        except Exception as e:
            import traceback
            error_msg = f"Ошибка создания GeoTIFF: {str(e)}"
            print(error_msg)
            print(traceback.format_exc())
            return None, error_msg