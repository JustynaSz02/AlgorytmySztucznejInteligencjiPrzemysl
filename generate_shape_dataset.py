import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Ellipse
import numpy as np
from PIL import Image
import os
import cv2
import io
import random
import shutil

# ============================================================================
# PARAMETRY GENEROWANIA DATASETU
# ============================================================================
NUM_TRAIN = 100  # Liczba obrazów na kategorię w zbiorze treningowym
NUM_VAL = 20     # Liczba obrazów na kategorię w zbiorze walidacyjnym
NUM_TEST = 20    # Liczba obrazów na kategorię w zbiorze testowym


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


def generate_background(width, height):
    """Generuje losowe tło: szum, gradient lub jednolity kolor"""
    bg_type = random.choice(['noise', 'gradient', 'solid', 'gradient_radial'])
    
    if bg_type == 'noise':
        # Szum gaussowski
        bg = np.random.normal(128, 30, (height, width, 3)).astype(np.uint8)
        bg = np.clip(bg, 0, 255)
    
    elif bg_type == 'gradient':
        # Gradient liniowy
        direction = random.choice(['horizontal', 'vertical', 'diagonal'])
        color1 = np.random.randint(0, 256, 3)
        color2 = np.random.randint(0, 256, 3)
        
        if direction == 'horizontal':
            bg = np.zeros((height, width, 3), dtype=np.uint8)
            for i in range(width):
                t = i / width
                bg[:, i] = (color1 * (1-t) + color2 * t).astype(np.uint8)
        elif direction == 'vertical':
            bg = np.zeros((height, width, 3), dtype=np.uint8)
            for i in range(height):
                t = i / height
                bg[i, :] = (color1 * (1-t) + color2 * t).astype(np.uint8)
        else:  # diagonal
            bg = np.zeros((height, width, 3), dtype=np.uint8)
            for i in range(height):
                for j in range(width):
                    t = (i + j) / (height + width)
                    bg[i, j] = (color1 * (1-t) + color2 * t).astype(np.uint8)
    
    elif bg_type == 'gradient_radial':
        # Gradient radialny
        center_x, center_y = random.uniform(0.3, 0.7) * width, random.uniform(0.3, 0.7) * height
        color1 = np.random.randint(0, 256, 3)
        color2 = np.random.randint(0, 256, 3)
        bg = np.zeros((height, width, 3), dtype=np.uint8)
        max_dist = np.sqrt(width**2 + height**2) / 2
        for i in range(height):
            for j in range(width):
                dist = np.sqrt((i - center_y)**2 + (j - center_x)**2)
                t = min(dist / max_dist, 1.0)
                bg[i, j] = (color1 * (1-t) + color2 * t).astype(np.uint8)
    
    else:  # solid
        # Jednolity kolor
        bg = np.full((height, width, 3), np.random.randint(0, 256, 3), dtype=np.uint8)
    
    return bg


def apply_augmentations(img, is_square=False):
    """Stosuje zaszumienie, zniekształcenia i inne transformacje do obrazu"""
    # Zaszumienie gaussowskie
    if random.random() < 0.5:
        noise = np.random.normal(0, random.uniform(5, 15), img.shape).astype(np.float32)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    
    # Zniekształcenia perspektywiczne - jednakowe dla kół i kwadratów
    if random.random() < 0.5:
        h, w = img.shape[:2]
        pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        offset = random.uniform(10, 25)  # Jednakowe zniekształcenia dla wszystkich
        
        pts2 = np.float32([
            [random.uniform(-offset, offset), random.uniform(-offset, offset)],
            [w + random.uniform(-offset, offset), random.uniform(-offset, offset)],
            [random.uniform(-offset, offset), h + random.uniform(-offset, offset)],
            [w + random.uniform(-offset, offset), h + random.uniform(-offset, offset)]
        ])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        img = cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    
    # Dodatkowe zniekształcenia affine (skalowanie, ścinanie) - dla kół i kwadratów
    if random.random() < 0.4:
        h, w = img.shape[:2]
        # Skalowanie nierównomierne
        scale_x = random.uniform(0.85, 1.15)
        scale_y = random.uniform(0.85, 1.15)
        # Ścinanie
        shear_x = random.uniform(-0.15, 0.15)
        shear_y = random.uniform(-0.15, 0.15)
        
        M = np.float32([
            [scale_x, shear_x, w * (1 - scale_x) / 2],
            [shear_y, scale_y, h * (1 - scale_y) / 2]
        ])
        img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    
    # Rozmycie (blur)
    if random.random() < 0.3:
        kernel_size = random.choice([3, 5])
        img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    
    # Zmiana jasności i kontrastu
    if random.random() < 0.4:
        alpha = random.uniform(0.8, 1.2)  # kontrast
        beta = random.uniform(-20, 20)     # jasność
        img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    
    return img


def generate_shape_image(color_code, color_name, shape_code, shape_name, 
                         rotation_range=(-180, 180), 
                         size_range_circle=(0.4, 0.8),
                         size_range_square=(0.4, 0.95),
                         figsize=4, dpi=224/4):
    """Generuje pojedynczy obraz kształtu z losowymi parametrami"""
    # Losowe parametry
    rotation = random.uniform(*rotation_range)
    # Różne zakresy rozmiarów dla kół i kwadratów
    if shape_code == 'round':
        shape_size = random.uniform(*size_range_circle)
    else:
        shape_size = random.uniform(*size_range_square)
    image_width = figsize
    shape_width = (image_width / 2) * shape_size
    axis_range = figsize / 2
    axis_min = -axis_range
    axis_max = axis_range
    
    # Tworzenie figury z tłem
    fig, ax = plt.subplots(figsize=(figsize, figsize))
    
    # Generowanie losowego tła
    img_width = int(figsize * dpi)
    img_height = int(figsize * dpi)
    background = generate_background(img_width, img_height)
    ax.imshow(background, extent=[axis_min, axis_max, axis_min, axis_max], zorder=0)
    
    # Obliczanie bezpiecznej strefy (margines) - figury nie mogą wychodzić poza kadr
    margin = shape_width * 0.6  # Margines bezpieczeństwa
    
    # Generowanie kształtu
    if shape_code == 'round':
        circle_radius = shape_width / 2
        # Bezpieczna strefa dla środka koła
        safe_min = axis_min + circle_radius + margin
        safe_max = axis_max - circle_radius - margin
        pos_x = random.uniform(safe_min, safe_max)
        pos_y = random.uniform(safe_min, safe_max)
        
        # Możliwość elipsy zamiast koła
        if random.random() < 0.3:
            width_ratio = random.uniform(0.8, 1.2)
            height_ratio = random.uniform(0.8, 1.2)
            poly = Ellipse(xy=(pos_x, pos_y), 
                          width=shape_width * width_ratio, 
                          height=shape_width * height_ratio, 
                          fc=color_code, alpha=random.uniform(0.5, 0.8))
        else:
            poly = Ellipse(xy=(pos_x, pos_y), width=shape_width, height=shape_width, 
                          fc=color_code, alpha=random.uniform(0.5, 0.8))
        ax.add_patch(poly)
    else:
        # Kwadrat z większymi deformacjami
        x, y = get_coords(shape_code)
        coords = np.c_[x, y]
        coords_normalized = coords / 20.0 * shape_width
        
        # Dodatkowe zniekształcenie kwadratu przed rotacją (trapez, równoległobok)
        if random.random() < 0.5:
            # Zniekształcenie trapezowe
            distortion = random.uniform(0.85, 1.15)
            coords_normalized[:, 0] *= distortion
            coords_normalized[:, 1] *= random.uniform(0.85, 1.15)
        
        coords_rotated = rotate_coords(coords_normalized, rotation)
        
        # Obliczanie maksymalnej odległości od środka (uwzględniając zniekształcenia)
        max_distance = np.max(np.linalg.norm(coords_rotated, axis=1)) + margin
        
        # Bezpieczna strefa dla środka kwadratu
        safe_min = axis_min + max_distance
        safe_max = axis_max - max_distance
        
        # Upewniamy się, że safe_min < safe_max
        if safe_min < safe_max:
            pos_x = random.uniform(safe_min, safe_max)
            pos_y = random.uniform(safe_min, safe_max)
        else:
            # Jeśli figura jest zbyt duża, centrujemy ją
            pos_x = 0
            pos_y = 0
        
        coords_final = coords_rotated + np.array([pos_x, pos_y])
        poly = Polygon(coords_final, fc=color_code, alpha=random.uniform(0.5, 0.8))
        ax.add_patch(poly)
    
    # Konfiguracja osi
    ax.axis('image')
    ax.axis('off')
    ax.set_xlim(axis_min, axis_max)
    ax.set_ylim(axis_min, axis_max)
    fig.tight_layout(pad=0)
    fig.canvas.draw()
    
    # Konwersja na obraz
    img = get_img_from_fig(fig, dpi=dpi)
    plt.close(fig)
    
    # Zastosowanie augmentacji (zaszumienie, zniekształcenia)
    is_square = (shape_code != 'round')
    img = apply_augmentations(img, is_square=is_square)
    
    return img


def generate_shape_dataset():
    """Generuje kompletny dataset z automatycznym czyszczeniem"""
    
    # Parametry
    figsize = 4
    dpi = 224 / figsize
    
    # Definicje kategorii
    colors = {'r': 'red', 'g': 'green', 'b': 'blue'}
    shapes = {4: 'square', 'round': 'circle'}
    
    # Czyszczenie całego folderu data
    base_dir = "data"
    if os.path.exists(base_dir):
        print(f"Usuwanie istniejącego folderu: {base_dir}")
        shutil.rmtree(base_dir)
    
    # Tworzenie struktury katalogów
    splits = ['train', 'val', 'test']
    counts = {'train': NUM_TRAIN, 'val': NUM_VAL, 'test': NUM_TEST}
    
    for split in splits:
        split_dir = os.path.join(base_dir, split)
        os.makedirs(split_dir, exist_ok=True)
        
        for color_code, color_name in colors.items():
            for shape_code, shape_name in shapes.items():
                category_name = f"{color_name}_{shape_name}"
                category_dir = os.path.join(split_dir, category_name)
                os.makedirs(category_dir, exist_ok=True)
                
                print(f"Generowanie {split}/{category_name}...")
                
                # Generowanie obrazów z różnymi zakresami parametrów
                for idx in range(counts[split]):
                    img = generate_shape_image(
                        color_code, color_name, shape_code, shape_name,
                        rotation_range=(-180, 180),      # Pełny zakres rotacji
                        size_range_circle=(0.4, 0.8),     # Rozmiar dla kół
                        size_range_square=(0.4, 0.95),    # Rozmiar dla kwadratów (większy maksimum)
                        figsize=figsize, dpi=dpi
                    )
                    
                    # Zapis obrazu
                    img_pil = Image.fromarray(img)
                    filename = f"{category_name}_{idx:04d}.png"
                    filepath = os.path.join(category_dir, filename)
                    img_pil.save(filepath)
                
                print(f"  ✓ {counts[split]} obrazów")
    
    total = (NUM_TRAIN + NUM_VAL + NUM_TEST) * len(colors) * len(shapes)
    print(f"\n✓ Dataset wygenerowany!")
    print(f"  Train: {NUM_TRAIN} x 6 kategorii = {NUM_TRAIN * 6} obrazów")
    print(f"  Val:   {NUM_VAL} x 6 kategorii = {NUM_VAL * 6} obrazów")
    print(f"  Test:  {NUM_TEST} x 6 kategorii = {NUM_TEST * 6} obrazów")
    print(f"  Łącznie: {total} obrazów")


if __name__ == "__main__":
    generate_shape_dataset()
