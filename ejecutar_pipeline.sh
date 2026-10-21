#!/bin/bash
echo "🚀 Iniciando Pipeline de Tres Lunas Serenas..."
python fuentes/datos_prep.py
python fuentes/train.py
echo "✅ Pipeline completado con éxito."
