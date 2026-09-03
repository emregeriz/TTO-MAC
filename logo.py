# -*- coding: utf-8 -*-
"""Trento logosu yardimcilari (camera_gui_v2.py'den tasindi)."""

from __future__ import annotations

import os

from PIL import Image


_TRENTO_LOGO_FILENAMES = ("trentoLogo.png", "trentoLogo.jpg")


def resolve_trento_logo_path():
    """Şeffaf arka plan için trentoLogo.png (öncelik), yoksa .jpg. Kamera_gui → repo → üst klasör."""
    base = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.dirname(base)
    for folder in (base, repo, os.path.normpath(os.path.join(repo, ".."))):
        for name in _TRENTO_LOGO_FILENAMES:
            p = os.path.join(folder, name)
            if os.path.isfile(p):
                return p
    return None


def _hex_to_rgb_triplet(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.strip().lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def pil_trento_logo_to_rgb(pil_img: Image.Image, bg_hex: str) -> Image.Image:
    """RGBA şeffaflığını bg_hex rengine birleştirir; CTkImage için RGB (siyah kutu yok)."""
    rgba = pil_img.convert("RGBA")
    bg = Image.new("RGBA", rgba.size, _hex_to_rgb_triplet(bg_hex) + (255,))
    bg.paste(rgba, (0, 0), rgba)
    return bg.convert("RGB")


# ============================================================================
#  MVS SDK Yardımcı Fonksiyonlar
# ============================================================================


