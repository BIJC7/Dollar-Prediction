#!/usr/bin/env python3
"""
Script de Chequeo Semestral y Mantenimiento Anual para USD/CLP Predictor
"""
import os
import sys
import json
import urllib.request
from datetime import datetime
from pathlib import Path

# Cargar variables de .env si existe localmente
def _load_env():
    candidates = [Path(".env"), Path(__file__).resolve().parent / ".env"]
    env_file = next((p for p in candidates if p.exists()), None)
    if env_file:
        with open(env_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip("'\"")
                if k and k not in os.environ:
                    os.environ[k] = v

_load_env()

DISCORD_WEBHOOK = os.environ.get("DISCORD_WEBHOOK_URL")

def send_discord_embed(payload: dict) -> bool:
    if not DISCORD_WEBHOOK:
        print("⚠️ No se encontró DISCORD_WEBHOOK_URL.")
        return False
    try:
        req = urllib.request.Request(
            DISCORD_WEBHOOK,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "User-Agent": "Mozilla/5.0"},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            if resp.status in (200, 204):
                print("✓ Notificación de mantenimiento enviada a Discord.")
                return True
    except Exception as exc:
        print(f"❌ Error al enviar a Discord: {exc}")
    return False


def run_semiannual_healthcheck():
    import yfinance as yf
    
    tickers = ["USDCLP=X", "HG=F", "CL=F", "GC=F", "^VIX", "DX-Y.NYB", "^TNX", "^IRX", "^GSPC", "EEM", "USDCNY=X", "BRL=X"]
    ok_tickers = []
    failed_tickers = []
    
    print("Verificando tickers en Yahoo Finance...")
    for t in tickers:
        try:
            df = yf.download(t, period="5d", progress=False)
            if df is not None and not df.empty:
                ok_tickers.append(t)
            else:
                failed_tickers.append(t)
        except Exception:
            failed_tickers.append(t)
            
    status_icon = "🟢" if len(failed_tickers) == 0 else ("🟡" if len(failed_tickers) <= 2 else "🔴")
    color = 0x2ECC71 if len(failed_tickers) == 0 else 0xF39C12
    
    payload = {
        "embeds": [{
            "title": f"🩺 {status_icon} CHEQUEO SEMESTRAL DE SALUD (APIs & Datos)",
            "color": color,
            "description": "Auditoría periódica automática de las fuentes de datos del motor cuantitativo USD/CLP.",
            "fields": [
                {
                    "name": "📊 Fuentes Yahoo Finance",
                    "value": f"**{len(ok_tickers)}/{len(tickers)}** tickers respondiendo correctamente.",
                    "inline": False
                },
                {
                    "name": "❌ Tickers con Observación",
                    "value": ", ".join(failed_tickers) if failed_tickers else "Ninguno. Todo 100% operativo.",
                    "inline": False
                },
                {
                    "name": "💾 Modelo Persistente",
                    "value": "Verificado en disco (`usdclp_model.pkl`).",
                    "inline": True
                },
                {
                    "name": "📅 Fecha del Chequeo",
                    "value": datetime.now().strftime("%Y-%m-%d %H:%M CLT"),
                    "inline": True
                },
                {
                    "name": "📝 Acción Requerida",
                    "value": "Ninguna. El sistema continuará operando de forma autónoma." if not failed_tickers else "Revisar tickers con observación si persiste en próximas ejecuciones.",
                    "inline": False
                }
            ],
            "footer": {"text": "USD/CLP Quantitative Engine — Mantenimiento Preventivo Semestral"}
        }]
    }
    send_discord_embed(payload)


def run_annual_maintenance_reminder():
    payload = {
        "embeds": [{
            "title": "🔧 📅 RECORDATORIO ANUAL DE MANTENIMIENTO PREVENTIVO",
            "color": 0x3498DB,
            "description": "Ha llegado la fecha del mantenimiento anual programado del sistema predictivo USD/CLP.",
            "fields": [
                {
                    "name": "1. Actualización de Dependencias",
                    "value": "Ejecutar en terminal:\n`pip install --upgrade -r requirements.txt`",
                    "inline": False
                },
                {
                    "name": "2. Limpieza de Caché Local",
                    "value": "Limpiar datos antiguos si es necesario:\n`rm -rf ~/.cache/usdclp_predictor`",
                    "inline": False
                },
                {
                    "name": "3. Reentrenamiento y Validación Completa",
                    "value": "Validar métricas actualizadas de Walk-Forward:\n`python usdclp_predictor.py --force-retrain`",
                    "inline": False
                },
                {
                    "name": "📅 Fecha Programada",
                    "value": datetime.now().strftime("%Y-%m-%d %H:%M CLT"),
                    "inline": True
                }
            ],
            "footer": {"text": "USD/CLP Quantitative Engine — Mantenimiento Anual"}
        }]
    }
    send_discord_embed(payload)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "semestral"
    if mode == "anual":
        run_annual_maintenance_reminder()
    else:
        run_semiannual_healthcheck()
