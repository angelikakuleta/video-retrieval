<#
.SYNOPSIS
  Ripping episodes from Blu-ray discs to lossless MKV (MakeMKV), one title at a time.

.USAGE
  .\rip-disc.ps1 -Action list                                   # list of titles on the disc
  .\rip-disc.ps1 -Output <folder> -Title <Nr> -Season S -Episode E   # one title -> sXXeYY.mkv
  .\rip-disc.ps1 -Output <folder> -Title <Nr>                        # one title -> title_<Nr>.mkv
  .\rip-disc.ps1 -Output <folder>                                    # all with Nr > MinLengthMin -> title_<Nr>.mkv
  -Prefix office   (optional) adds a prefix to the file name: office_s01e02.mkv, office_title_7.mkv

  Nr = number of the .mpls playlist ("Title N" in Leawo). Titles without Nr (.m2ts, .mpls(1)) are not ripped.
#>

param(
  [ValidateSet('rip','list')]
  [string]$Action = 'rip',
  [string]$Output = '',
  [string]$Prefix = '',
  [int]$Title = -1,
  [int]$Season = 0,
  [int]$Episode = 0,
  [int]$Disc = 0,
  [int]$MinLengthMin = 15
)

$ErrorActionPreference = 'Stop'

$mkv = @(
  "C:\Program Files (x86)\MakeMKV\makemkvcon64.exe",
  "C:\Program Files\MakeMKV\makemkvcon64.exe"
) | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $mkv) { throw "makemkvcon64.exe not found. Install MakeMKV: https://www.makemkv.com/download/" }

function Convert-DurToSec([string]$d) {
  $p = $d.Trim() -split ':'
  if ($p.Count -eq 3) { return [int]$p[0]*3600 + [int]$p[1]*60 + [int]$p[2] }
  return -1
}

function Get-Titles {
  $raw = & $mkv -r --cache=1 --minlength=1 info "disc:$Disc" 2>$null
  $titles = @{}
  foreach ($line in $raw) {
    if ($line -match '^DRV:(\d+),(\d+),\d+,\d+,"([^"]*)","([^"]*)"' -and [int]$Matches[2] -ne 256) {
      Write-Host ("DRIVE {0}: {1}  [disc: {2}]" -f $Matches[1], $Matches[3], $Matches[4])
    }
    if ($line -match '^TINFO:(\d+),(\d+),\d+,"(.*)"$') {
      $id = [int]$Matches[1]; $code = [int]$Matches[2]; $val = $Matches[3]
      if (-not $titles.ContainsKey($id)) { $titles[$id] = [ordered]@{ Id = $id; Nr = $null; Seconds = -1 } }
      switch ($code) {
        9  { $titles[$id].Duration = $val; $titles[$id].Seconds = Convert-DurToSec $val }
        8  { $titles[$id].Chapters = $val }
        10 { $titles[$id].Size = $val }
        16 { $titles[$id].Source = $val
             if ($val -match '^(\d+)\.mpls$') { $titles[$id].Nr = [int]$Matches[1] } }
      }
    }
  }
  return ($titles.Values | Sort-Object { $_.Id })
}

function New-FileName([string]$base) {
  if ($Prefix) { return "${Prefix}_$base.mkv" } else { return "$base.mkv" }
}

function Invoke-RipTitle($t, [string]$TargetName) {
  $target = Join-Path $Output $TargetName
  Write-Host ("Nr {0} (Id {1}, {2}) -> {3}" -f $t.Nr, $t.Id, $t.Duration, $TargetName)
  if (Test-Path $target) { Write-Host "  skipping - file already exists"; return }

  $before = @(Get-ChildItem $Output -Filter *.mkv | Select-Object -ExpandProperty Name)
  & $mkv -r --minlength=1 mkv "disc:$Disc" $t.Id $Output | Out-Null
  if ($LASTEXITCODE -ne 0) { Write-Warning "makemkvcon returned code $LASTEXITCODE"; return }

  $new = @(Get-ChildItem $Output -Filter *.mkv | Where-Object { $before -notcontains $_.Name })
  if ($new.Count -eq 0) { Write-Warning "No new .mkv file was created"; return }
  Move-Item $new[0].FullName $target
  Write-Host "  -> $target"
}

# ---------------- LIST ----------------
if ($Action -eq 'list') {
  Write-Host "Reading the disc in drive $Disc ..."
  $t = Get-Titles
  if (-not $t) { Write-Host "No titles. Check that the disc is in the drive and the -Disc index."; return }
  $t | ForEach-Object {
    [pscustomobject]@{ Nr = $_.Nr; Id = $_.Id; Duration = $_.Duration; Chapters = $_.Chapters; Size = $_.Size; Source = $_.Source }
  } | Format-Table -AutoSize
  Write-Host "Ripping: .\rip-disc.ps1 -Output <folder> -Title <Nr> -Season <season> -Episode <episode>"
  return
}

# ---------------- RIP ----------------
if (-not $Output) { throw "Specify the target folder: -Output <path>" }
if (($Season -gt 0) -xor ($Episode -gt 0)) { throw "Specify -Season and -Episode together or neither of them." }
if (($Season -gt 0) -and ($Title -lt 0)) { throw "With -Season/-Episode also specify -Title <Nr>." }
if (-not (Test-Path $Output)) { New-Item -ItemType Directory -Force -Path $Output | Out-Null }

Write-Host "Reading the disc in drive $Disc ..."
$all = @(Get-Titles)
if ($all.Count -eq 0) { throw "No titles on the disc. Check that the disc is in the drive and the -Disc index." }

if ($Title -ge 0) {
  $t = @($all | Where-Object { $_.Nr -eq $Title })
  if ($t.Count -ne 1) { throw "Nr ${Title}: found $($t.Count) titles. Check: .\rip-disc.ps1 -Action list" }
  $name = if ($Season -gt 0) { "s{0:D2}e{1:D2}" -f $Season, $Episode } else { "title_$Title" }
  Invoke-RipTitle $t[0] (New-FileName $name)
}
else {
  $sel = @($all | Where-Object { $null -ne $_.Nr -and $_.Seconds -ge $MinLengthMin * 60 } | Sort-Object { $_.Nr })
  if ($sel.Count -eq 0) { throw "No playlists longer than $MinLengthMin min." }
  Write-Host "Ripping $($sel.Count) playlist(s) to: $Output"
  foreach ($t in $sel) { Invoke-RipTitle $t (New-FileName "title_$($t.Nr)") }
}
Write-Host "Done. Files in: $Output"
