@echo off
cd /d "%~dp0"
python -X utf8 -m ray_tracing.runtime.cli %*
