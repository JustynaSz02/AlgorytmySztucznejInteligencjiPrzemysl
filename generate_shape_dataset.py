import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Ellipse
import numpy as np
from PIL import Image
import os
import cv2
import io
import random


def get_img_from_fig(fig, dpi=None):
    """Konwertuje matplotlib figure na obraz numpy array"""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def get_coords(n):
    """Generuje współrzędne dla wielokąta o n bokach"""
    t = np.arange(0, 360 + (360 / n), 360 / n)
    x = 10 * np.sin(np.radians(t))
    y = 10 * np.cos(np.radians(t))
    return x, y


def rotate_coords(coords, angle_degrees):
    """Obraca współrzędne o podany kąt w stopniach"""
    angle_rad = np.radians(angle_degrees)
    rotation_matrix = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad)],
        [np.sin(angle_rad), np.cos(angle_rad)]
    ])
    return coords @ rotation_matrix.T


def generate_shape_images():
    """Generuje dataset kształtów: 6 kategorii x 50 obrazów"""
    
    # Parametry
    figsize = 4  # Rozmiar w calach
    dpi = 224 / figsize
    alpha = 0.6
    num_images_per_category = 10
    
    # Szerokość obrazu w jednostkach matplotlib (odpowiednik figsize)
    image_width = figsize
    
    # Szerokość kształtu = 1/4 szerokości obrazu
    shape_width = image_width / 2
    
    # Zakres osi - ustawiamy tak, aby obraz był wycentrowany
    axis_range = figsize / 2
    axis_min = -axis_range
    axis_max = axis_range
    
    # Definicje kategorii
    colors = {
        'r': 'red',      # czerwony
        'g': 'green',    # zielony
        'b': 'blue'      # niebieski
    }
    
    shapes = {
        4: 'square',     # kwadrat
        'round': 'circle'  # koło
    }
    
    # Główny folder na obrazy
    output_dir = "data/test"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Generowanie obrazów dla każdej kategorii
    for color_code, color_name in colors.items():
        for shape_code, shape_name in shapes.items():
            # Tworzymy folder dla kategorii (np. "red_circle", "green_square")
            category_name = f"{color_name}_{shape_name}"
            category_dir = os.path.join(output_dir, category_name)
            
            if not os.path.exists(category_dir):
                os.makedirs(category_dir)
            
            print(f"Generowanie obrazów dla: {category_name}")
            
            # Generujemy 50 obrazów z losowymi parametrami
            for idx in range(num_images_per_category):
                # Losowa rotacja z zakresu -90 do +90 stopni
                rotation = random.uniform(-90, 90)
                
                # Tworzymy figurę
                fig, ax = plt.subplots(figsize=(figsize, figsize))
                
                # Generujemy kształt
                if shape_code == 'round':
                    # Koło: szerokość = shape_width
                    # Promień koła w jednostkach współrzędnych
                    circle_radius = shape_width / 2
                    
                    # Maksymalna pozycja środka, aby koło nie wychodziło poza obszar
                    max_pos = axis_max - circle_radius
                    min_pos = axis_min + circle_radius
                    
                    # Losowa pozycja środka koła
                    pos_x = random.uniform(min_pos, max_pos)
                    pos_y = random.uniform(min_pos, max_pos)
                    
                    # Tworzymy koło
                    poly = Ellipse(xy=(pos_x, pos_y), width=shape_width, height=shape_width, 
                                 fc=color_code, alpha=alpha)
                    ax.add_patch(poly)
                else:
                    # Kwadrat (n=4) z losową rotacją i pozycją
                    x, y = get_coords(shape_code)
                    coords = np.c_[x, y]
                    
                    # Normalizujemy współrzędne kwadratu (obecnie są w zakresie ~-10 do 10)
                    # Chcemy, aby kwadrat miał szerokość shape_width
                    # Współrzędne z get_coords mają rozmiar około 20 (od -10 do 10)
                    # Więc skalujemy do shape_width
                    coords_normalized = coords / 20.0 * shape_width
                    
                    # Obracamy współrzędne
                    coords_rotated = rotate_coords(coords_normalized, rotation)
                    
                    # Obliczamy maksymalne przesunięcie kwadratu (w przypadku rotacji do 45 stopni)
                    # Dla kwadratu o szerokości shape_width, przekątna po rotacji może być sqrt(2) * shape_width
                    # Więc maksymalna odległość od środka do wierzchołka to sqrt(2) * shape_width / 2
                    max_distance = (shape_width / 2) * np.sqrt(2)
                    
                    # Maksymalna pozycja środka, aby kwadrat nie wychodził poza obszar
                    max_pos = axis_max - max_distance
                    min_pos = axis_min + max_distance
                    
                    # Losowa pozycja środka kwadratu
                    pos_x = random.uniform(min_pos, max_pos)
                    pos_y = random.uniform(min_pos, max_pos)
                    
                    # Przesuwamy do losowej pozycji
                    coords_final = coords_rotated + np.array([pos_x, pos_y])
                    
                    poly = Polygon(coords_final, fc=color_code, alpha=alpha)
                    ax.add_patch(poly)
                
                # Konfiguracja osi - ustawiamy zakres tak, aby obraz był wycentrowany
                ax.axis('image')
                ax.axis('off')
                ax.set_xlim(axis_min, axis_max)
                ax.set_ylim(axis_min, axis_max)
                
                # Tight layout
                fig.tight_layout(pad=0)
                fig.canvas.draw()
                
                # Konwersja na obraz
                img = get_img_from_fig(fig, dpi=dpi)
                plt.close(fig)  # Zamknij figurę, aby zwolnić pamięć
                
                # Zapisujemy obraz
                img_pil = Image.fromarray(img)
                filename = f"{category_name}_{idx:03d}.png"
                filepath = os.path.join(category_dir, filename)
                img_pil.save(filepath)
            
            print(f"  ✓ Wygenerowano {num_images_per_category} obrazów dla {category_name}")
    
    print(f"\n✓ Dataset wygenerowany w folderze: {output_dir}")
    print(f"  Łącznie: 6 kategorii x {num_images_per_category} obrazów = {6 * num_images_per_category} obrazów")


if __name__ == "__main__":
    generate_shape_images()

