#!/usr/bin/env bash
set -Eeuo pipefail

# ------------------------------------------------------------
# Eisenbahn App - robustes Startscript
# - Erstellt/prüft venv (Python 3.12)
# - Installiert Requirements
# - GPU Detection (NVIDIA) + Fallback auf CPU
# - Startet app_eisenbahn.py
# ------------------------------------------------------------

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
cd "$SCRIPT_DIR"

APP_PY="${APP_PY:-app_eisenbahn.py}"
VENV_DIR="${VENV_DIR:-.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3.12}"

# Optional: Port/Host (dein Python setzt port=5001 fix; hier nur für Info/Später)
APP_HOST="${APP_HOST:-0.0.0.0}"
APP_PORT="${APP_PORT:-5001}"

# Defaults: Auto-Device Auswahl; kann über ENV überschrieben werden:
# export EISENBAHN_DEVICE=cpu | cuda:0
EISENBAHN_DEVICE="${EISENBAHN_DEVICE:-auto}"

log()  { printf '[%s] %s\n' "INFO" "$*"; }
warn() { printf '[%s] %s\n' "WARN" "$*" >&2; }
err()  { printf '[%s] %s\n' "ERROR" "$*" >&2; }
die()  { err "$*"; exit 1; }

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "Fehlendes Kommando: $1"
}

pyver_ok() {
  # returns 0 if python is >= 3.12
  "$1" - <<'PY' >/dev/null 2>&1
import sys
ok = (sys.version_info.major, sys.version_info.minor) >= (3,12)
raise SystemExit(0 if ok else 1)
PY
}

ensure_python() {
  if command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    if pyver_ok "$PYTHON_BIN"; then
      log "Python OK: $("$PYTHON_BIN" -V)"
      return 0
    else
      die "$PYTHON_BIN gefunden, aber nicht >= 3.12: $("$PYTHON_BIN" -V)"
    fi
  fi

  # Fallback: python3, aber nur wenn >= 3.12
  if command -v python3 >/dev/null 2>&1 && pyver_ok python3; then
    PYTHON_BIN="python3"
    log "Nutze Fallback Python: $(python3 -V)"
    return 0
  fi

  die "Python 3.12 nicht gefunden. Installiere Python 3.12 oder setze PYTHON_BIN=/pfad/zu/python3.12"
}

ensure_venv() {
  ensure_python

  # Prüfe venv Fähigkeit
  "$PYTHON_BIN" - <<'PY' >/dev/null 2>&1 || die "python venv Modul fehlt (python3.12-venv installieren?)."
import venv
PY

  if [[ -d "$VENV_DIR" && -x "$VENV_DIR/bin/python" ]]; then
    log "venv vorhanden: $VENV_DIR"
  else
    log "Erzeuge venv: $VENV_DIR"
    "$PYTHON_BIN" -m venv "$VENV_DIR"
  fi

  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"

  # pip/wheel/setuptools robust aktualisieren
  python -m pip install --upgrade pip setuptools wheel >/dev/null
  log "pip: $(python -m pip --version)"
}

detect_gpu_and_set_device() {
  # Wenn User explizit gesetzt hat, nicht überschreiben
  if [[ "$EISENBAHN_DEVICE" != "auto" ]]; then
    export EISENBAHN_DEVICE
    log "Device per ENV erzwungen: EISENBAHN_DEVICE=$EISENBAHN_DEVICE"
    return 0
  fi

  # NVIDIA tools optional
  if command -v nvidia-smi >/dev/null 2>&1; then
    if nvidia-smi -L >/dev/null 2>&1; then
      # Torch CUDA Check im venv
      if python - <<'PY' >/dev/null 2>&1
import torch
raise SystemExit(0 if torch.cuda.is_available() else 1)
PY
      then
        export EISENBAHN_DEVICE="cuda:0"
        log "NVIDIA GPU erkannt + torch.cuda verfügbar -> EISENBAHN_DEVICE=cuda:0"
        return 0
      else
        warn "NVIDIA GPU erkannt, aber torch.cuda ist NICHT verfügbar -> CPU Fallback"
      fi
    else
      warn "nvidia-smi vorhanden, aber keine GPU gelistet -> CPU"
    fi
  else
    log "nvidia-smi nicht gefunden -> CPU"
  fi

  export EISENBAHN_DEVICE="cpu"
  log "EISENBAHN_DEVICE=cpu"
}

install_python_requirements() {
  # Wenn requirements.txt existiert, nutze es (präferiert)
  if [[ -f "requirements.txt" ]]; then
    log "Installiere requirements.txt"
    python -m pip install -r requirements.txt
    return 0
  fi

  # Sonst minimaler Satz für dein Script (ultralytics bringt torch-Abhängigkeiten, aber CUDA-Wheels sind systemabhängig)
  log "Keine requirements.txt gefunden -> installiere Minimal-Dependencies"
  python -m pip install --upgrade \
    flask \
    opencv-python \
    numpy \
    ultralytics \
    pyyaml \
    paho-mqtt
}

sanity_checks() {
  [[ -f "$APP_PY" ]] || die "Python App nicht gefunden: $APP_PY"

  # ffmpeg ist für RTSP in vielen Setups wichtig (OpenCV FFMPEG Backend)
  if command -v ffmpeg >/dev/null 2>&1; then
    log "ffmpeg vorhanden: $(ffmpeg -version 2>/dev/null | head -n1)"
  else
    warn "ffmpeg nicht gefunden. RTSP kann je nach OpenCV-Build trotzdem gehen, oft aber nicht. Empfehlung: ffmpeg installieren."
  fi

  # Quick import check
  python - <<PY
import sys
mods = ["cv2","numpy","flask","ultralytics","torch"]
missing=[]
for m in mods:
    try:
        __import__(m)
    except Exception as e:
        missing.append((m,str(e)))
if missing:
    print("Missing/Problematic imports:")
    for m,e in missing:
        print(f" - {m}: {e}")
    sys.exit(1)
print("Import check OK.")
PY
}

print_runtime_info() {
  log "Arbeitsverzeichnis: $SCRIPT_DIR"
  log "App: $APP_PY"
  log "venv: $VENV_DIR"
  log "Host/Port (Info): $APP_HOST:$APP_PORT"
  log "EISENBAHN_DEVICE: ${EISENBAHN_DEVICE:-unset}"
  python - <<'PY'
import torch, sys
print(f"Python: {sys.version.split()[0]}")
print(f"Torch:  {getattr(torch,'__version__','?')}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    try:
        print("CUDA device:", torch.cuda.get_device_name(0))
    except Exception as e:
        print("CUDA device name error:", e)
PY
}

main() {
  need_cmd bash
  ensure_venv
  install_python_requirements
  sanity_checks
  detect_gpu_and_set_device
  print_runtime_info

  log "Starte Anwendung..."
  exec python "$APP_PY"
}

main "$@"
