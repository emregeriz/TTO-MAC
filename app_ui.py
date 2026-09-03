"""
PaddleOCR PP-OCRv5 — fotoğraf yükleme arayüzü (Gradio).

Çalıştır (klasör içindeyken):
  python app_ui.py
  python app_ui.py --server-port 7860 --device cpu
"""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from typing import Any, TYPE_CHECKING

# gradio is only needed for the web UI. The desktop GUI (ocr_desktop_native.py)
# imports this module solely for `_get_ocr`. Keep gradio as a lazy import so
# users without gradio installed can still run the desktop app.
if TYPE_CHECKING:
    import gradio as gr  # noqa: F401

import ocr_engine


def _format_ocr_error(err: BaseException) -> str:
    msg = str(err)
    parts: list[str] = [f"**Hata:** `{msg}`"]
    low = msg.lower()
    if "winerror 127" in low or "shm.dll" in low or ("torch" in low and ("load" in low or "dll" in low)):
        parts.append(
            "**Windows — torch / DLL:** Bu mesaj çoğunlukla eksik **Microsoft Visual C++ 2015–2022 (x64)** "
            "çalışma zamanı veya bozuk bir **PyTorch** kurulumundan kaynaklanır (mobile model şart değil)."
        )
        parts.append(
            "1. [VC++ Redistributable x64](https://aka.ms/vs/17/release/vc_redist.x64.exe) kurun, gerekirse PC’yi yeniden başlatın.\n"
            "2. Hâlâ olmazsa PyTorch’u temizleyip CPU sürümünü yeniden kurun:\n"
            "`pip uninstall -y torch torchvision torchaudio`\n"
            "`pip install torch --index-url https://download.pytorch.org/whl/cpu`"
        )
    return "\n\n".join(parts)


def _ensure_image_path(image: Any) -> str | None:
    if image is None:
        return None
    if isinstance(image, str) and os.path.isfile(image):
        return image
    try:
        from PIL import Image as PILImage

        if isinstance(image, PILImage.Image):
            fd, path = tempfile.mkstemp(suffix=".png")
            os.close(fd)
            image.convert("RGB").save(path)
            return path
    except Exception:
        pass
    try:
        import numpy as np

        if isinstance(image, np.ndarray):
            import cv2

            fd, path = tempfile.mkstemp(suffix=".png")
            os.close(fd)
            bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(path, bgr)
            return path
    except Exception:
        pass
    return None


_ocr_cache: dict[str, Any] = {"key": None, "ocr": None}


def _get_ocr(device: str, mobile: bool, use_v4: bool, lang: str) -> Any:
    if ocr_engine.paddleocr_major_version() >= 3:
        ver: str | None = "PP-OCRv4" if use_v4 else "PP-OCRv5"
    else:
        ver = "PP-OCRv4" if use_v4 else None
    lang_clean = (lang or "").strip() or None
    key = (device, mobile, ver, lang_clean)
    if _ocr_cache["key"] != key:
        kw = ocr_engine.ocr_kwargs(
            device=device,
            mobile=mobile,
            ocr_version=ver,
            lang=lang_clean,
        )
        _ocr_cache["ocr"] = ocr_engine.build_ocr(**kw)
        _ocr_cache["key"] = key
    return _ocr_cache["ocr"]


def run_ocr(
    image: Any,
    device: str,
    mobile: bool,
    use_v4: bool,
    lang: str,
) -> tuple[Any, str, str, str]:
    path = _ensure_image_path(image)
    if not path:
        return None, "", "", "Önce bir görüntü yükleyin."

    prev_dir = _ocr_cache.get("_last_viz_dir")
    if isinstance(prev_dir, str) and os.path.isdir(prev_dir):
        shutil.rmtree(prev_dir, ignore_errors=True)

    try:
        ocr = _get_ocr(device, mobile, use_v4, lang)
        out = ocr_engine.predict_with_details(ocr, path)
        _ocr_cache["_last_viz_dir"] = out.get("_viz_dir")

        viz = out.get("viz_path")
        if viz and os.path.isfile(viz):
            ann = viz
        else:
            ann = path

        ocr_label = (
            "PP-OCRv4"
            if use_v4
            else ("PP-OCRv5" if ocr_engine.paddleocr_major_version() >= 3 else "varsayılan (2.x)")
        )
        info = (
            f"**Satır sayısı:** {len(out['texts'])}\n\n"
            f"**Ayarlar:** cihaz=`{device}`, mobile={mobile}, OCR={ocr_label}, dil={lang or 'varsayılan'}"
        )
        return ann, out["markdown_table"], out["full_text"], info
    except Exception as e:
        _ocr_cache["key"] = None
        _ocr_cache["ocr"] = None
        return None, "", "", _format_ocr_error(e)


def build_ui(default_device: str):
    import gradio as gr  # lazy: web UI only

    with gr.Blocks(title="PaddleOCR — PP-OCRv5") as demo:
        gr.Markdown(
            "## PaddleOCR — fotoğraftan metin ve rakam\n"
            "Görüntü yükleyin; algılanan satırlar tabloda listelenir. "
            "İlk çalıştırmada modeller indirilebilir (biraz sürebilir).\n\n"
            "**Windows:** `torch` / `shm.dll` / WinError 127 görürseniz aşağıdaki **VC++ Redistributable x64** "
            "kurulumunu yapın; çoğu ortamda sorun böyle çözülür."
        )
        with gr.Row():
            with gr.Column(scale=1):
                img_in = gr.Image(type="numpy", label="Görüntü", image_mode="RGB")
                device = gr.Dropdown(
                    choices=["cpu", "gpu", "gpu:0"],
                    value=default_device,
                    label="Cihaz",
                )
                mobile = gr.Checkbox(label="PP-OCRv5 mobile modelleri", value=False)
                use_v4 = gr.Checkbox(
                    label="PP-OCRv4 kullan (PaddleOCR 3.x kuruluysa varsayılan PP-OCRv5 yerine)",
                    value=False,
                )
                lang = gr.Textbox(label="Dil (isteğe bağlı, örn. en)", placeholder="Boş = varsayılan")
                btn = gr.Button("OCR çalıştır", variant="primary")
            with gr.Column(scale=1):
                img_out = gr.Image(label="Kutu / sonuç görseli")
                info = gr.Markdown()
        tbl = gr.Markdown(label="Sonuç tablosu")
        raw = gr.Textbox(label="Düz metin (tüm satırlar)", lines=12)

        btn.click(
            fn=run_ocr,
            inputs=[img_in, device, mobile, use_v4, lang],
            outputs=[img_out, tbl, raw, info],
        )
    return demo


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cpu", help="Varsayılan Paddle cihazı")
    p.add_argument("--server-name", default="127.0.0.1")
    p.add_argument("--server-port", type=int, default=7860)
    args = p.parse_args()

    demo = build_ui(default_device=args.device)
    demo.launch(server_name=args.server_name, server_port=args.server_port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
