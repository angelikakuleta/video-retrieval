<#
.SYNOPSIS
  Normalise video files to the format used by the pipeline:
  MP4 / H.264 (CRF 18) / constant framerate / keyframe every 48 frames / no audio.
  Constant framerate and fixed keyframes make frame seeking deterministic.

.PARAMETER InputPath
  A video file, or a folder whose *.mkv / *.mp4 files (no subfolders) are all processed.
.PARAMETER OutputPath
  A folder (output keeps the input name with .mp4), or - for a single input file -
  a full path ending with .mp4.
.PARAMETER FFmpeg
  Path to ffmpeg.exe. By default the conda environment's copy is used, then PATH.
.PARAMETER Force
  Overwrite outputs that already exist (by default they are skipped).

.EXAMPLE
  .\normalize-video.ps1 -InputPath C:\video-retrieval\data\raw\tbbt -OutputPath C:\video-retrieval\data\processed\tbbt
  .\normalize-video.ps1 -InputPath D:\rip\title_t00.mkv -OutputPath C:\video-retrieval\data\processed\office\office_s01e02.mp4
#>

param(
  [Parameter(Mandatory = $true)] [string]$InputPath,
  [Parameter(Mandatory = $true)] [string]$OutputPath,
  [string]$FFmpeg,
  [switch]$Force
)

$ErrorActionPreference = 'Stop'
$extensions = @('.mkv', '.mp4')

# FFmpeg comes from the conda environment, and conda puts it in <env>\Library\bin,
# which lands on PATH only after `conda activate`. Look there first, exactly as
# src/utils/tools.py does for the Python side, so the script also runs from a
# plain PowerShell window.
$candidates = @()
if ($FFmpeg) { $candidates += $FFmpeg }
if ($env:CONDA_PREFIX) { $candidates += (Join-Path $env:CONDA_PREFIX 'Library\bin\ffmpeg.exe') }
$candidates += (Join-Path $env:USERPROFILE 'miniforge3\envs\wideo\Library\bin\ffmpeg.exe')
$candidates += (Join-Path $env:USERPROFILE 'miniconda3\envs\wideo\Library\bin\ffmpeg.exe')

$ffmpeg = $candidates | Where-Object { $_ -and (Test-Path $_) } | Select-Object -First 1
if (-not $ffmpeg) { $ffmpeg = (Get-Command ffmpeg -ErrorAction SilentlyContinue).Source }
if (-not $ffmpeg) {
  throw ("ffmpeg not found. Activate the environment (conda activate wideo), " +
         "pass -FFmpeg <path>, or install it: " +
         'conda install -n wideo -c conda-forge "ffmpeg=*=gpl*"')
}
if (-not (Test-Path $InputPath)) { throw "Input path does not exist: $InputPath" }

# input: one file or every video file directly in the folder
$inItem = Get-Item $InputPath
if ($inItem.PSIsContainer) {
  $files = @(Get-ChildItem $inItem.FullName -File | Where-Object { $extensions -contains $_.Extension.ToLower() } | Sort-Object Name)
  if ($files.Count -eq 0) { Write-Host "No video files in $($inItem.FullName)"; return }
} else {
  $files = @($inItem)
}

# output: explicit file name only for a single input file, otherwise a folder
$outIsFile = if (Test-Path $OutputPath) { -not (Get-Item $OutputPath).PSIsContainer }
             else { [IO.Path]::GetExtension($OutputPath).ToLower() -eq '.mp4' }
if ($outIsFile -and $files.Count -ne 1) { throw "OutputPath is a file name but InputPath is a folder" }
$outDir = if ($outIsFile) { Split-Path ([IO.Path]::GetFullPath($OutputPath)) -Parent } else { [IO.Path]::GetFullPath($OutputPath) }
if (-not (Test-Path $outDir)) { New-Item -ItemType Directory -Path $outDir | Out-Null }

$i = 0
foreach ($f in $files) {
  $i++
  $out = if ($outIsFile) { [IO.Path]::GetFullPath($OutputPath) }
         else { Join-Path $outDir ([IO.Path]::GetFileNameWithoutExtension($f.Name) + '.mp4') }
  $tag = "[$i/$($files.Count)] $($f.Name)"

  if ($out -ieq $f.FullName) { Write-Warning "$tag -> would overwrite the input, skipped"; continue }
  if ((Test-Path $out) -and -not $Force) { Write-Host "$tag -> exists, skipped"; continue }

  Write-Host "$tag -> $out"
  $timer = [Diagnostics.Stopwatch]::StartNew()
  $ErrorActionPreference = 'Continue'   # ffmpeg writes to stderr; only real errors get through -loglevel error
  & $ffmpeg -y -hide_banner -loglevel error -i $f.FullName `
      -map 0:v:0 -an -sn `
      -c:v libx264 -preset slow -crf 18 -pix_fmt yuv420p `
      -fps_mode cfr -g 48 -keyint_min 48 -sc_threshold 0 `
      -movflags +faststart $out 2>&1 | ForEach-Object { Write-Host "  $_" }
  $code = $LASTEXITCODE
  $ErrorActionPreference = 'Stop'
  if ($code -ne 0) { Write-Warning "$tag -> ffmpeg failed (code $code)"; continue }
  Write-Host ("  done in {0:mm\:ss}" -f $timer.Elapsed)
}
