@echo off
echo Starting full indexing...
curl.exe -X POST "http://localhost:8000/ingest/all"
pause