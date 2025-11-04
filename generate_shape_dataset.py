import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Ellipse
import numpy as np
from PIL import Image
import os, io, cv2, random, argparse

# ------------------------ Narzędzia ------------------------

def get_img_from_fig(fig, dpi=None):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def rotate(coords, deg):
    a = np.radians(deg)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    return coords @ R.T

def shear(coords, shx=0.0, shy=0.0):
    S = np.array([[1, shx],[shy, 1]])
    return coords @ S.T

def scale_xy(coords, sx=1.0, sy=1.0):
    S = np.array([[sx,0],[0,sy]])
    return coords @ S.T

def square_coords(side=1.0):
    # kwadrat jednostkowy (zamknięty)
    h = side/2
    pts = np.array([[-h,-h],[h,-h],[h,h],[-h,h]])
    return pts

def random_bg(h, w):
    """Zwraca tło (h,w,3) typu uint8.
       Losuje typ: solid-light / solid-dark / solid-color / gradient / noise
    """
    choice = random.choice(["solid_light","solid_dark","solid_color","gradient","noise"])
    if choice=="solid_light":
        v = random.randint(210,255)
        bg = np.full((h,w,3), v, np.uint8)
    elif choice=="solid_dark":
        v = random.randint(0,45)
        bg = np.full((h,w,3), v, np.uint8)
    elif choice=="solid_color":
        bg = np.array([random.randint(80,255),
                       random.randint(80,255),
                       random.randint(80,255)], np.uint8)
        bg = np.full((h,w,3), bg, np.uint8)
    elif choice=="gradient":
        x = np.linspace(0,1,w, dtype=np.float32)
        c1 = np.array([random.randint(0,255) for _ in range(3)], dtype=np.float32)
        c2 = np.array([random.randint(0,255) for _ in range(3)], dtype=np.float32)
        row = (c1[None,:]*(1-x[:,None]) + c2[None,:]*x[:,None]).astype(np.uint8)
        bg = np.repeat(row[None,:,:], h, axis=0)
    else:  # noise
        bg = np.random.randint(0,255,(h,w,3), dtype=np.uint8)
        k = random.choice([3,5,7])
        bg = cv2.GaussianBlur(bg, (k,k), sigmaX=0)
    return bg

def paste_alpha(base_rgb, overlay_rgb, mask_alpha):
    """Łączy overlay (RGB) z bazą (RGB) wg maski alfa w [0,1]."""
    a = mask_alpha[...,None]
    out = (overlay_rgb*a + base_rgb*(1-a)).astype(np.uint8)
    return out

# ------------------------ Generator ------------------------

def generate_dataset(per_class=100, splits=(70,15,15), out_dir="data",
                     img_size=224, seed=42):
    random.seed(seed); np.random.seed(seed)
    assert sum(splits)==100, "splits muszą sumować się do 100 (np. 70 15 15)"

    # Klasy: 3 kolory × 2 kształty
    color_map = {"red":(255,0,0),"green":(0,170,0),"blue":(0,120,255)}
    shapes = ["circle","square"]
    classes = [f"{c}_{s}" for c in color_map for s in shapes]

    # Przygotuj katalogi
    for split_name in ["train","val","test"]:
        for cls in classes:
            os.makedirs(os.path.join(out_dir, split_name, cls), exist_ok=True)

    # Ile obrazów na split
    n_train = per_class * splits[0] // 100
    n_val   = per_class * splits[1] // 100
    n_test  = per_class - n_train - n_val
    counts = {"train":n_train, "val":n_val, "test":n_test}

    # Parametry figury (rysujemy matplotlib → PNG → numpy)
    figsize = 4
    dpi = img_size/figsize
    axis_min, axis_max = -figsize/2, figsize/2
    shape_base = figsize/2    # bazowa szerokość kształtu

    for cls in classes:
        color_name, shape_name = cls.split("_")
        color_rgb = color_map[color_name]

        print(f"[{cls}] train={counts['train']} val={counts['val']} test={counts['test']}")

        for split_name, n_imgs in counts.items():
            for i in range(n_imgs):
                # 1) rysujemy tło jako imshow
                fig, ax = plt.subplots(figsize=(figsize, figsize))
                ax.axis("off"); ax.set_xlim(axis_min,axis_max); ax.set_ylim(axis_min,axis_max)

                # generuj tło i narysuj
                bg = random_bg(img_size, img_size)
                ax.imshow(bg, extent=(axis_min,axis_max,axis_min,axis_max))

                # 2) kształt + losowe zniekształcenia
                alpha = random.uniform(0.7, 1.0)

                if shape_name=="circle":
                    # elipsa (zniekształcony „okrąg”)
                    sx = shape_base * random.uniform(0.35, 0.55)   # średnica x
                    sy = shape_base * random.uniform(0.35, 0.55)   # średnica y
                    rot = random.uniform(0, 180)

                    # bezpieczne pozycjonowanie (część krawędzi może wyjść minimalnie – wygląda naturalniej)
                    margin = max(sx, sy)/2 + 0.2
                    cx = random.uniform(axis_min+margin, axis_max-margin)
                    cy = random.uniform(axis_min+margin, axis_max-margin)

                    patch = Ellipse((cx,cy), width=sx, height=sy,
                                    angle=rot, fc=np.array(color_rgb)/255, ec=None, alpha=alpha)
                    ax.add_patch(patch)

                else:  # square
                    side = shape_base * random.uniform(0.55, 0.75)
                    pts = square_coords(side)

                    # jitter wierzchołków (do 10% boku)
                    jitter = side*0.10
                    pts = pts + np.random.uniform(-jitter, jitter, pts.shape)

                    # skala niesymetryczna i shear
                    sx = random.uniform(0.85, 1.15)
                    sy = random.uniform(0.85, 1.15)
                    shx = random.uniform(-0.25, 0.25)
                    shy = random.uniform(-0.25, 0.25)
                    rot = random.uniform(-90, 90)

                    pts = scale_xy(pts, sx, sy)
                    pts = shear(pts, shx, shy)
                    pts = rotate(pts, rot)

                    # bezpieczny margines dla przesunięcia
                    extent = np.max(np.linalg.norm(pts, axis=1))
                    margin = extent + 0.2
                    cx = random.uniform(axis_min+margin, axis_max-margin)
                    cy = random.uniform(axis_min+margin, axis_max-margin)

                    pts = pts + np.array([cx, cy])
                    patch = Polygon(pts, closed=True, fc=np.array(color_rgb)/255, ec=None, alpha=alpha)
                    ax.add_patch(patch)

                # 3) czasem mała „niedoskonałość” (szum/rozmazanie całości)
                fig.canvas.draw()
                img = get_img_from_fig(fig, dpi=dpi)
                plt.close(fig)

                # 10% – szum Gaussa
                if random.random() < 0.10:
                    noise = np.random.normal(0, 10, img.shape).astype(np.int16)
                    img = np.clip(img.astype(np.int16)+noise, 0, 255).astype(np.uint8)

                # 15% – lekki blur
                if random.random() < 0.15:
                    k = random.choice([3,5])
                    img = cv2.GaussianBlur(img, (k,k), 0)

                # 4) zapis
                cls_dir = os.path.join(out_dir, split_name, cls)
                fname = f"{cls}_{i:04d}.png"
                Image.fromarray(img).save(os.path.join(cls_dir, fname))

    print("\n✓ Gotowe! Zbiory zapisane w:", os.path.abspath(out_dir))

# ------------------------ CLI ------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Generator datasetu kształtów z losowymi zniekształceniami i tłami.")
    ap.add_argument("--per-class", type=int, default=100, help="liczba obrazów na klasę (łącznie train+val+test)")
    ap.add_argument("--splits", nargs=3, type=int, default=[70,15,15], metavar=("TRAIN","VAL","TEST"),
                    help="procentowe podziały (muszą sumować się do 100)")
    ap.add_argument("--out", type=str, default="data", help="katalog wyjściowy (powstaną podfoldery train/val/test)")
    ap.add_argument("--img-size", type=int, default=224, help="rozmiar obrazu (kwadrat)")
    ap.add_argument("--seed", type=int, default=42, help="ziarno losowe")
    args = ap.parse_args()

    generate_dataset(per_class=args.per_class, splits=tuple(args.splits),
                     out_dir=args.out, img_size=args.img_size, seed=args.seed)
