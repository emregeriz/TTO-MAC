#!/usr/bin/env bash
# TTO-MAC kurulum — macOS
# Hicbir sey kurulu olmayan bir Mac'te tek komutla calisir:
#     chmod +x kur.sh && ./kur.sh
set -u

KOK="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$KOK"

bilgi() { printf "\n\033[1;32m==> %s\033[0m\n" "$*"; }
uyari() { printf "\033[1;33m[UYARI] %s\033[0m\n" "$*"; }
hata()  { printf "\033[1;31m[HATA] %s\033[0m\n" "$*"; }

bilgi "TTO-MAC kurulumu basliyor"
echo "Klasor : $KOK"
echo "Sistem : $(sw_vers -productName 2>/dev/null) $(sw_vers -productVersion 2>/dev/null) / $(uname -m)"

# ---------------------------------------------------------------- 1) Xcode CLT
if ! xcode-select -p >/dev/null 2>&1; then
  bilgi "Xcode Command Line Tools kuruluyor (pencere acilabilir, bitince bu betigi TEKRAR calistirin)"
  xcode-select --install || true
  echo "Kurulum penceresi acildiysa bitmesini bekleyip ./kur.sh komutunu tekrar calistirin."
  exit 0
fi
echo "Xcode CLT: tamam"

# ------------------------------------------------------------------ 2) Homebrew
if ! command -v brew >/dev/null 2>&1; then
  bilgi "Homebrew kuruluyor (sifre isteyebilir)"
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)" || {
    hata "Homebrew kurulamadi. https://brew.sh adresinden elle kurup tekrar deneyin."; exit 1; }
fi
# Apple Silicon'da brew /opt/homebrew, Intel'de /usr/local
for BREW in /opt/homebrew/bin/brew /usr/local/bin/brew; do
  [ -x "$BREW" ] && eval "$("$BREW" shellenv)" && break
done
command -v brew >/dev/null 2>&1 || { hata "brew PATH'te yok"; exit 1; }
echo "Homebrew: $(brew --version | head -1)"

# --------------------------------------------------- 3) Python 3.11 + Tk (ONEMLI)
# Homebrew'un python'u tkinter'i AYRI paketle veriyor; python-tk olmadan
# arayuz hic acilmaz ("No module named _tkinter").
bilgi "Python 3.11 ve Tk kuruluyor"
brew list python@3.11 >/dev/null 2>&1 || brew install python@3.11
brew list python-tk@3.11 >/dev/null 2>&1 || brew install python-tk@3.11

PY="$(brew --prefix python@3.11)/bin/python3.11"
[ -x "$PY" ] || PY="$(command -v python3.11 || true)"
[ -x "$PY" ] || { hata "python3.11 bulunamadi"; exit 1; }
echo "Python : $PY  ($("$PY" -V 2>&1))"

if ! "$PY" -c "import tkinter" >/dev/null 2>&1; then
  hata "tkinter calismiyor. 'brew install python-tk@3.11' komutunu elle deneyin."
  exit 1
fi
echo "tkinter: tamam"

# ------------------------------------------------------------ 4) Sanal ortam
bilgi "Sanal ortam (.venv) hazirlaniyor"
[ -d .venv ] || "$PY" -m venv .venv
PIP=".venv/bin/pip"
"$PIP" install --upgrade pip wheel setuptools >/dev/null

# ------------------------------------------------------- 5) Zorunlu paketler
bilgi "Uygulama paketleri kuruluyor (birkac dakika surebilir)"
"$PIP" install -r requirements.txt || { hata "Zorunlu paketler kurulamadi"; exit 1; }

# ------------------------------------------------- 6) OCR (opsiyonel, buyuk)
# PaddleOCR olmadan da uygulama calisir; yalniz "sistem tespiti" (okunamayan
# kasanin uzerindeki numaradan seri okuma) devre disi kalir.
bilgi "OCR (PaddleOCR) kuruluyor — opsiyonel, basarisiz olursa uygulama yine calisir"
if "$PIP" install -r requirements-ocr.txt; then
  echo "OCR: kuruldu"
else
  uyari "OCR kurulamadi. Sayim ve barkod calisir; otomatik seri tespiti kapali olur."
fi

# ------------------------------------------------------------- 7) Dogrulama
bilgi "Kurulum dogrulaniyor"
.venv/bin/python - <<'PYEOF'
import importlib, platform, sys
print(f"Python {sys.version.split()[0]}  {platform.machine()}")
zorunlu = ["tkinter", "customtkinter", "cv2", "PIL", "numpy", "ultralytics",
           "torch", "zxingcpp"]
eksik = []
for m in zorunlu:
    try:
        importlib.import_module(m)
        print(f"  [OK]   {m}")
    except Exception as ex:
        eksik.append(m)
        print(f"  [EKSIK] {m}: {ex}")
try:
    import paddleocr  # noqa: F401
    print("  [OK]   paddleocr (OCR acik)")
except Exception:
    print("  [BILGI] paddleocr yok — OCR kapali, sayim yine calisir")
try:
    import torch
    mps = getattr(torch.backends, "mps", None)
    print("  Apple GPU (MPS):", "VAR" if (mps and mps.is_available()) else "yok")
except Exception:
    pass
import os
model = os.path.join("models", "V8LAST.pt")
print(f"  YOLO modeli: {'VAR' if os.path.isfile(model) else 'YOK'} ({model})")
gorsel = len([f for f in os.listdir("ornek_goruntuler")
              if f.lower().endswith((".jpg", ".png"))]) if os.path.isdir("ornek_goruntuler") else 0
print(f"  Ornek goruntu: {gorsel} adet")
sys.exit(1 if eksik else 0)
PYEOF
DURUM=$?

chmod +x baslat.sh 2>/dev/null || true
if [ $DURUM -eq 0 ]; then
  bilgi "KURULUM TAMAM"
  echo "Baslatmak icin:  ./baslat.sh"
else
  hata "Bazi zorunlu paketler eksik — yukaridaki [EKSIK] satirlarina bakin."
  exit 1
fi
