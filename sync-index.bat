@echo off
echo Synchronizing index with moved or removed books...
curl.exe -X POST "http://localhost:8000/books/sync-index"
pause
