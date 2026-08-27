#!/usr/bin/env bash
# =============================================================================
# USD/CLP Predictor v5.0 — Script de Ejecución y Alertas Automáticas
# =============================================================================

# Definir directorio del proyecto
PROJECT_DIR="/home/yo/Documentos/Predicción del dólar"
cd "$PROJECT_DIR" || exit 1

# Ruta al intérprete de Python del entorno virtual
PYTHON_BIN="$PROJECT_DIR/venv_usdclp/bin/python"
SCRIPT_PATH="$PROJECT_DIR/usdclp_predictor.py"
LOG_FILE="$PROJECT_DIR/usdclp_execution.log"

echo "==================================================" >> "$LOG_FILE"
echo "Ejecución iniciada: $(date '+%Y-%m-%d %H:%M:%S')" >> "$LOG_FILE"
echo "==================================================" >> "$LOG_FILE"

# Ejecutar el predictor con notificaciones activadas
"$PYTHON_BIN" "$SCRIPT_PATH" --notify >> "$LOG_FILE" 2>&1
EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "Ejecución finalizada con éxito: $(date '+%Y-%m-%d %H:%M:%S')" >> "$LOG_FILE"
else
    echo "ERROR en la ejecución (Código $EXIT_CODE): $(date '+%Y-%m-%d %H:%M:%S')" >> "$LOG_FILE"
fi

exit $EXIT_CODE
