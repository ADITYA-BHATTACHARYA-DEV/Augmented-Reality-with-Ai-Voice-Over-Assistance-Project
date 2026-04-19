@echo off
echo ================================================
echo  AR Car v4 — Ollama Model Setup
echo ================================================
echo.
echo This script pulls the two models needed:
echo   llava   = vision model (sees your screen)
echo   llama3  = fast text model (answers questions)
echo.
echo Make sure ollama is installed: https://ollama.com
echo.

ollama pull llava-phi3
ollama pull llama3

echo.
echo ================================================
echo  Done! Now run in two terminals:
echo  Terminal 1:  ollama serve
echo  Terminal 2:  python ar_car_v4.py
echo ================================================
pause