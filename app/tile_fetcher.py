import requests
from io import BytesIO
from PIL import Image
import math

class TileFetcher:
    """Простой загрузчик тайлов с OSM/ArcGIS"""
    
    @staticmethod
    def deg2num(lat_deg, lon_deg, zoom):
        """Перевод координат в номер тайла"""
        lat_rad = math.radians(lat_deg)
        n = 2.0 ** zoom
        xtile = int((lon_deg + 180.0) / 360.0 * n)
        ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
        return xtile, ytile
    
    @staticmethod
    def fetch_tile(xtile, ytile, zoom, source='osm'):
        """Скачать один тайл"""
        if source == 'osm':
            url = f"https://tile.openstreetmap.org/{zoom}/{xtile}/{ytile}.png"
        elif source == 'arcgis':
            url = f"https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{zoom}/{ytile}/{xtile}"
        else:
            raise ValueError(f"Unknown source: {source}")
        
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return Image.open(BytesIO(response.content))
    
    @staticmethod
    def get_area_tiles(lat, lon, width_px, height_px, zoom=18):
        """
        Получить мозаику тайлов для области вокруг точки
        Возвращает PIL Image и координаты углов (lat, lon)
        """
        # Вычисляем количество тайлов в ширину и высоту
        # Примерно 256px на тайл при zoom=18
        tiles_x = max(2, math.ceil(width_px / 256) + 1)
        tiles_y = max(2, math.ceil(height_px / 256) + 1)
        
        # Центральный тайл
        center_x, center_y = TileFetcher.deg2num(lat, lon, zoom)
        
        # Собираем мозаику
        mosaic = Image.new('RGB', (tiles_x * 256, tiles_y * 256))
        
        for i in range(tiles_x):
            for j in range(tiles_y):
                tile_x = center_x - (tiles_x // 2) + i
                tile_y = center_y - (tiles_y // 2) + j
                
                try:
                    tile = TileFetcher.fetch_tile(tile_x, tile_y, zoom, source='arcgis')
                    mosaic.paste(tile, (i * 256, j * 256))
                except Exception as e:
                    print(f"Failed to fetch tile {tile_x},{tile_y}: {e}")
                    # Заполняем черным если тайл не загрузился
                    mosaic.paste((0, 0, 0), (i * 256, j * 256, (i+1)*256, (j+1)*256))
        
        # Вычисляем координаты углов мозаики
        def num2deg(xtile, ytile, zoom):
            n = 2.0 ** zoom
            lon_deg = xtile / n * 360.0 - 180.0
            lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
            lat_deg = math.degrees(lat_rad)
            return lat_deg, lon_deg
        
        left = center_x - (tiles_x // 2)
        top = center_y - (tiles_y // 2)
        right = left + tiles_x
        bottom = top + tiles_y
        
        top_left_lat, top_left_lon = num2deg(left, top, zoom)
        bottom_right_lat, bottom_right_lon = num2deg(right, bottom, zoom)
        
        # Обрезаем до нужных размеров (центрируем)
        crop_x = (mosaic.width - width_px) // 2
        crop_y = (mosaic.height - height_px) // 2
        cropped = mosaic.crop((
            crop_x, crop_y, 
            crop_x + width_px, 
            crop_y + height_px
        ))
        
        # Корректируем координаты после обрезки
        pix_per_lon = (bottom_right_lon - top_left_lon) / mosaic.width
        pix_per_lat = (top_left_lat - bottom_right_lat) / mosaic.height
        
        new_top_left_lon = top_left_lon + (crop_x * pix_per_lon)
        new_top_left_lat = top_left_lat - (crop_y * pix_per_lat)
        new_bottom_right_lon = new_top_left_lon + (width_px * pix_per_lon)
        new_bottom_right_lat = new_top_left_lat - (height_px * pix_per_lat)
        
        bounds = {
            'top_left': (new_top_left_lat, new_top_left_lon),
            'bottom_right': (new_bottom_right_lat, new_bottom_right_lon)
        }
        
        return cropped, bounds