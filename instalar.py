#!/usr/bin/env python3
"""
Instalación Simple - Generador de Datos Sintéticos con GAN
"""

import subprocess
import sys
import os

def main():
    print("🚀 INSTALACIÓN SIMPLE - GENERADOR DE DATOS SINTÉTICOS")
    print("=" * 60)
    
    print("\n📦 Instalando dependencias...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencias instaladas")
    except Exception as e:
        print(f"❌ Error: {e}")
        return
    
    print("\n📁 Creando directorios...")
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('data/synthetic', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    print("✅ Directorios creados")
    
    print("\n🎉 INSTALACIÓN COMPLETADA")
    print("=" * 60)
    print("\n🚀 Para usar:")
    print("   python main.py --data datos_ejemplo.csv --target target")
    print("\n📊 Con tus datos:")
    print("   python main.py --data tu_archivo.csv --target columna_objetivo")

if __name__ == "__main__":
    main()
