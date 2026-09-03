#!/usr/bin/env bash
# TTO-MAC baslatici
set -u
KOK="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$KOK"

if [ ! -x ".venv/bin/python" ]; then
  echo "[HATA] .venv yok. Once kurulumu calistirin:  ./kur.sh"
  exit 1
fi
exec .venv/bin/python app.py
