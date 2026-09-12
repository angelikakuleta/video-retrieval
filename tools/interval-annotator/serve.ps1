# Serves this folder over http, so that "Export CSV" can ask where to save.
#
# Chrome only hands the File System Access API (the save dialog) to a page with
# a normal origin. Opened straight from disk as file://, the tool still works --
# the export just lands in the downloads folder instead of asking.
#
#     .\serve.ps1              # http://localhost:8765
#     .\serve.ps1 -Port 9000
#
# Ctrl+C stops it. Any Python will do; the project environment is only a default.

param(
    [int]$Port = 8765,
    [string]$Python = "C:\Users\PC\miniforge3\envs\wideo\python.exe"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $Python)) {
    $found = Get-Command python -ErrorAction SilentlyContinue
    if (-not $found) { throw "No Python found. Pass one with -Python <path to python.exe>." }
    $Python = $found.Source
}

$root = $PSScriptRoot
Write-Host "Interval Annotator -> http://localhost:$Port  (Ctrl+C to stop)"
Start-Process "http://localhost:$Port"
& $Python -m http.server $Port --directory $root
