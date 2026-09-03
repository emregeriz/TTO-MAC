# -*- coding: utf-8 -*-
"""YOLO kasa tespiti + barkod okuma — macOS sürümü.

Windows sürümündeki `Kamera_gui/camera_gui_v2.DetectionProcessor` ile AYNI
çıktı sözleşmesini sunar; tek fark Hikrobot MVS SDK'sına hiç dokunmamasıdır
(o SDK yalnız Windows/Linux içindir).

Barkod motoru: **zxing-cpp**. Windows'taki ücretli Aremak Code Reader bir
.NET DLL + USB dongle gerektirdiği için macOS'ta çalışmaz; bu sürüm açık
kaynaklı zxing-cpp kullanır. Okuma oranı Aremak'a göre bir miktar düşük
olabilir — bu port HIZ ÖLÇÜMÜ ve arayüz denemesi içindir.
"""

from __future__ import annotations

import os
import platform

from datetime import datetime

import cv2
import numpy as np
from PIL import Image, ImageTk  # noqa: F401  (cizim yardimcilari icin)

try:
    from ultralytics import YOLO
    import torch

    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False
    YOLO = None
    torch = None
    print("[UYARI] Ultralytics/YOLO bulunamadi. pip install ultralytics")

try:
    import zxingcpp as zx

    BARCODE_AVAILABLE = True
    BARCODE_ERROR = None
except ImportError as _ex:
    BARCODE_AVAILABLE = False
    zx = None
    BARCODE_ERROR = str(_ex)
    print("[UYARI] zxing-cpp bulunamadi. pip install zxing-cpp")


def torch_device():
    """macOS'ta Apple Silicon GPU (MPS), yoksa CPU. CUDA varsa o.

    Ultralytics `predict(device=...)` için uygun değeri döner.
    """
    if torch is None:
        return "cpu"
    try:
        if torch.cuda.is_available():
            return 0
        # Apple Silicon: Metal Performance Shaders
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available() and platform.system() == "Darwin":
            return "mps"
    except Exception:
        pass
    return "cpu"


# ---------------------------------------------------------------- model yolu
_MODEL_BASENAME = "V8LAST.pt"


def _candidate_yolo_model_paths():
    """models/ klasörü ve proje kökünde .pt modelini ara."""
    _dir = os.path.dirname(os.path.abspath(__file__))
    return [
        os.path.join(_dir, "models", _MODEL_BASENAME),
        os.path.join(_dir, _MODEL_BASENAME),
    ]


def resolve_yolo_model_path():
    override = os.environ.get("ODAI_YOLO_MODEL", "").strip()
    if override and os.path.isfile(override):
        return override
    for p in _candidate_yolo_model_paths():
        if os.path.isfile(p):
            return p
    # models/ içindeki HERHANGİ bir .pt dosyasını kabul et
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    if os.path.isdir(models_dir):
        for name in sorted(os.listdir(models_dir)):
            if name.lower().endswith(".pt"):
                return os.path.join(models_dir, name)
    return None


CONFIDENCE_THRESHOLD = 0.65
TARGET_CLASS_NAME = None
CROP_EXT = ".jpg"


def preprocess_crop_for_barcode(crop_bgr):
    """
    Renkli kasa kırpmasını barkod okumaya uygun gri görüntüye çevirir:
    gri tonlama, CLAHE kontrast, hafif keskinleştirme; küçük kırpmaları sınırlı ölçüde büyütür.
    """
    if crop_bgr is None or getattr(crop_bgr, "size", 0) == 0:
        return None
    if len(crop_bgr.shape) == 3 and crop_bgr.shape[2] >= 3:
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    elif len(crop_bgr.shape) == 2:
        gray = np.ascontiguousarray(crop_bgr, dtype=np.uint8)
    else:
        return None

    h, w = gray.shape[:2]
    if h < 2 or w < 2:
        return gray

    min_edge = min(h, w)
    target_min = 360
    scale = 1.0
    if min_edge < target_min:
        scale = float(target_min) / float(min_edge)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    max_side = max(nh, nw, 1)
    if max_side > 2400:
        scale *= 2400.0 / float(max_side)
        nh, nw = int(round(h * scale)), int(round(w * scale))
    if nh != h or nw != w:
        interp = cv2.INTER_CUBIC if nh > h else cv2.INTER_AREA
        gray = cv2.resize(gray, (max(1, nw), max(1, nh)), interpolation=interp)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blur = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=1.2)
    sharpened = cv2.addWeighted(enhanced, 1.45, blur, -0.45, 0.0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


class DetectionProcessor:
    """YOLO kasa tespiti ve barkod okuma işlemcisi."""

    def __init__(self):
        self.model = None
        self.model_names = {}

    def load_model(self, log_fn=None):
        if not HAS_YOLO:
            raise Exception("Ultralytics/YOLO yüklü değil! pip install ultralytics")

        if self.model is None:
            if log_fn:
                log_fn("YOLO modeli yükleniyor...")
            model_path = resolve_yolo_model_path()
            if not model_path:
                aranan = "\n".join(f"  • {p}" for p in _candidate_yolo_model_paths())
                raise FileNotFoundError(
                    "YOLO .pt modeli bulunamadı. Dosyayı şu yollardan birine koyun:\n" + aranan
                )
            if log_fn:
                log_fn(f"Model: {os.path.basename(model_path)}")
            self.model = YOLO(model_path)
            device = torch_device()
            if device != "cpu":
                self.model.to("cuda" if device == 0 else device)
            if log_fn:
                isim = {0: "GPU (CUDA)", "mps": "Apple GPU (MPS)"}.get(device, "CPU")
                log_fn(f"{isim} kullanılıyor.")
            self.model_names = getattr(self.model, "names", {})
            if log_fn:
                log_fn("Model yüklendi.")

    def process_image(self, img_bgr, save_dir, log_fn=None):
        self.load_model(log_fn)

        def out(msg):
            if log_fn:
                log_fn(msg)

        os.makedirs(save_dir, exist_ok=True)
        crops_dir = os.path.join(save_dir, "crops")
        os.makedirs(crops_dir, exist_ok=True)

        device = torch_device()

        out("YOLO tespiti yapılıyor...")
        results = self.model.predict(source=img_bgr, save=False, conf=CONFIDENCE_THRESHOLD, device=device)

        if not results:
            out("YOLO sonuç döndürmedi!")
            return {
                "toplam_kasa": 0,
                "okunan_barkod": 0,
                "okunamayan_barkod": 0,
                "annotated_path": None,
                "annotated_image": None,
                "barcode_preprocess_image": None,
                "barcode_preprocess_path": None,
                "barkod_bulunamayan_isimler": [],
                "barkod_bulunamayan_yollar": [],
                "kasalar": [],
                "image_width": int(img_bgr.shape[1]) if img_bgr is not None else 0,
                "image_height": int(img_bgr.shape[0]) if img_bgr is not None else 0,
                "barkod_icerikler": [],
            }

        result = results[0]
        crate_count = 0
        boxes = getattr(result, "boxes", None)
        if boxes is not None and getattr(boxes, "cls", None) is not None:
            cls_array = boxes.cls.cpu().numpy()
            if TARGET_CLASS_NAME:
                for cls_idx in cls_array:
                    if str(self.model_names.get(int(cls_idx), "")).lower() == TARGET_CLASS_NAME.lower():
                        crate_count += 1
            else:
                crate_count = len(cls_array)

        out(f"Tespit edilen kasa sayısı: {crate_count}")

        toplam_barkod = 0
        barkod_bulunamayan_isimler = []
        barkod_bulunamayan_yollar = []
        box_barkod_durumu = {}
        barkod_icerikler = []
        kasalar = []  # Kasa başına detay: {bbox, barkodlar, barkod_okundu}

        orig_img = getattr(result, "orig_img", img_bgr)
        if len(orig_img.shape) == 3 and orig_img.shape[2] >= 3:
            g0 = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)
        else:
            g0 = np.asarray(orig_img, dtype=np.uint8)
        barcode_vis = cv2.cvtColor(g0, cv2.COLOR_GRAY2BGR)

        annotated = orig_img.copy()

        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

        if boxes is not None and hasattr(boxes, "xyxy"):
            xyxy = boxes.xyxy.cpu().numpy()
            for i, (x1, y1, x2, y2) in enumerate(xyxy):
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                h, w = orig_img.shape[:2]
                x1, x2 = max(0, x1), min(w, x2)
                y1, y2 = max(0, y1), min(h, y2)
                if x2 > x1 and y2 > y1:
                    crop_color = orig_img[y1:y2, x1:x2].copy()
                    proc_hi = preprocess_crop_for_barcode(crop_color)
                    pw, ph = x2 - x1, y2 - y1
                    if proc_hi is not None:
                        fit = cv2.resize(proc_hi, (pw, ph), interpolation=cv2.INTER_AREA)
                        barcode_vis[y1:y2, x1:x2] = cv2.cvtColor(fit, cv2.COLOR_GRAY2BGR)
                    else:
                        fit = cv2.resize(cv2.cvtColor(crop_color, cv2.COLOR_BGR2GRAY), (pw, ph), interpolation=cv2.INTER_AREA)
                        barcode_vis[y1:y2, x1:x2] = cv2.cvtColor(fit, cv2.COLOR_GRAY2BGR)

                    kasa_no = i + 1
                    crop_name = f"{timestamp_str}_kasa_{kasa_no}{CROP_EXT}"
                    barkod_okundu = True
                    icerikler = []

                    if BARCODE_AVAILABLE:
                        zx_src = proc_hi if proc_hi is not None else cv2.cvtColor(crop_color, cv2.COLOR_BGR2GRAY)
                        barcodes = zx.read_barcodes(zx_src)
                        n_barcode = len(barcodes)
                        if n_barcode == 0:
                            barkod_okundu = False
                            crop_name = f"OKUNAMADI_{timestamp_str}_kasa_{kasa_no}{CROP_EXT}"
                            crop_full_path = os.path.join(crops_dir, crop_name)
                            barkod_bulunamayan_isimler.append(crop_name)
                            barkod_bulunamayan_yollar.append(crop_full_path)
                            out(f"  Barkod bulunamadı: Kasa #{kasa_no}")
                        else:
                            toplam_barkod += n_barcode
                            icerikler = [b.text for b in barcodes]
                            barkod_icerikler.extend(icerikler)
                            out(f"  Kasa #{kasa_no} barkod: {n_barcode} adet → {icerikler}")
                    else:
                        out(f"  Kasa #{kasa_no} (barkod okuma devre dışı)")

                    box_barkod_durumu[i] = barkod_okundu
                    kasalar.append({
                        "kasa_no": kasa_no,
                        "bbox": (x1, y1, x2, y2),
                        "barkodlar": list(icerikler),
                        "barkod_okundu": barkod_okundu,
                    })
                    crop_path = os.path.join(crops_dir, crop_name)
                    cv2.imwrite(crop_path, crop_color)
            if len(xyxy) == 0:
                pf0 = preprocess_crop_for_barcode(orig_img)
                if pf0 is not None:
                    barcode_vis = cv2.cvtColor(pf0, cv2.COLOR_GRAY2BGR)
        else:
            pf = preprocess_crop_for_barcode(orig_img)
            if pf is not None:
                barcode_vis = cv2.cvtColor(pf, cv2.COLOR_GRAY2BGR)

        if boxes is not None and hasattr(boxes, "xyxy"):
            xyxy = boxes.xyxy.cpu().numpy()

            COLOR_OK = (255, 255, 0)
            COLOR_FAIL = (0, 0, 255)
            for i, (x1, y1, x2, y2) in enumerate(xyxy):
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                okundu = box_barkod_durumu.get(i, True)
                color = COLOR_OK if okundu else COLOR_FAIL
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)

                kasa_no = i + 1
                numara = str(kasa_no)
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.7
                thickness = 2
                (tw, th), _ = cv2.getTextSize(numara, font, font_scale, thickness)
                tx = x2 - tw - 4
                ty = y1 + th + 4
                cv2.rectangle(
                    annotated,
                    (tx - 2, ty - th - 2),
                    (tx + tw + 2, ty + 2),
                    (255, 255, 255),
                    -1,
                )
                cv2.putText(annotated, numara, (tx, ty), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

        annotated_path = os.path.join(save_dir, f"annotated_{timestamp_str}.jpg")
        cv2.imwrite(annotated_path, annotated)

        barcode_prep_path = os.path.join(save_dir, f"barcode_preprocess_{timestamp_str}.jpg")
        cv2.imwrite(barcode_prep_path, barcode_vis)

        out("")
        out("=" * 50)
        out("ÖZET")
        out("=" * 50)
        out(f"Toplam kasa: {crate_count}")
        out(f"Okunan barkod: {toplam_barkod}")
        out(f"Okunamayan barkod: {len(barkod_bulunamayan_isimler)}")
        if barkod_icerikler:
            out(f"Barkod içerikleri: {barkod_icerikler}")
        out("=" * 50)

        image_h, image_w = orig_img.shape[:2]
        return {
            "toplam_kasa": crate_count,
            "okunan_barkod": toplam_barkod,
            "okunamayan_barkod": len(barkod_bulunamayan_isimler),
            "annotated_path": annotated_path,
            "annotated_image": annotated,
            "barcode_preprocess_image": barcode_vis,
            "barcode_preprocess_path": barcode_prep_path,
            "barkod_bulunamayan_isimler": barkod_bulunamayan_isimler,
            "barkod_bulunamayan_yollar": barkod_bulunamayan_yollar,
            "kasalar": kasalar,
            "image_width": int(image_w),
            "image_height": int(image_h),
            "barkod_icerikler": list(barkod_icerikler),
        }


# ============================================================================
#  Yardımcı GUI Fonksiyonları
# ============================================================================


def _fit_image(pil_img, max_w, max_h):
    w, h = pil_img.size
    ratio = min(max_w / w, max_h / h, 1.0)
    return pil_img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)


def _fit_image_allow_upscale(pil_img, max_w, max_h):
    """Görüntüyü max_w x max_h alanına sığdırır; gerekirse büyütür (zoom için)."""
    w, h = pil_img.size
    if w < 1 or h < 1:
        return pil_img
    ratio = min(max_w / float(w), max_h / float(h))
    return pil_img.resize((max(1, int(round(w * ratio))), max(1, int(round(h * ratio)))), Image.LANCZOS)


def numpy_to_photoimage(np_img, max_w, max_h):
    """BGR (veya tek kanal gri) numpy görüntüyü Tkinter PhotoImage'e çevirir."""
    if np_img is None:
        return None
    if len(np_img.shape) == 2:
        rgb = cv2.cvtColor(np_img, cv2.COLOR_GRAY2RGB)
    else:
        rgb = cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    pil_img = _fit_image(pil_img, max_w, max_h)
    return ImageTk.PhotoImage(pil_img)


def numpy_to_photoimage_viewport(np_img, max_w, max_h):
    """Kırpılmış görüntüyü alana sığdırır; küçük kırpma büyütülebilir."""
    if np_img is None:
        return None
    if len(np_img.shape) == 2:
        rgb = cv2.cvtColor(np_img, cv2.COLOR_GRAY2RGB)
    else:
        rgb = cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    pil_img = _fit_image_allow_upscale(pil_img, max_w, max_h)
    return ImageTk.PhotoImage(pil_img)


# ============================================================================
#  MODERN STAT CARD WİDGET
# ============================================================================


