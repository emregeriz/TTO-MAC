# -*- coding: utf-8 -*-
"""macOS'ta kamera yöneticisi vekili.

Hikrobot MVS SDK yalnız Windows ve Linux için dağıtılıyor; macOS sürümü yok.
Bu yüzden Mac portunda **canlı kamera bağlantısı yoktur** — uygulama görüntü
dosyalarından beslenir (her kamera kartındaki **Yükle** düğmesi ya da araç
çubuğundaki **6 GÖRSEL YÜKLE**).

Bu sınıf `multi_camera_gui_2.MultiCameraManager` ile aynı arayüzü sunar ama
hiçbir cihaz bulmaz; böylece arayüz kodu platformdan bağımsız kalır ve
"Kameraları Tara" düğmesi çökmek yerine anlamlı bir mesaj verir.
"""

from __future__ import annotations

import platform


class MvsUnavailable(RuntimeError):
    """MVS SDK bu platformda yok."""


class MultiCameraManager:
    """Cihaz bulmayan, arayüzü bozmayan vekil."""

    #: Arayüzün operatöre gösterdiği açıklama.
    REASON = (
        "macOS'ta canlı kamera desteği yok (Hikrobot MVS SDK yalnızca "
        "Windows/Linux). Görüntüleri Yükle düğmeleriyle verin."
    )

    def __init__(self):
        self.cameras = []
        self.sdk_ready = False

    def initialize_sdk(self):
        # Sessizce başarısız olur: "Kameraları Tara" sonrası 0 cihaz bulunur
        # ve arayüz zaten "0/6 bağlı" yazar.
        self.sdk_ready = False
        return False

    def enum_devices(self):
        return []

    def connect_selected(self, infos):
        self.cameras = []
        return []

    def disconnect_all(self):
        self.cameras = []

    def finalize_sdk(self):
        self.sdk_ready = False

    @staticmethod
    def platform_note() -> str:
        return f"{platform.system()} {platform.machine()} — {MultiCameraManager.REASON}"
