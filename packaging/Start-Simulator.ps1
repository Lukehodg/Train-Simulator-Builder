# Train Link Simulator - zero-dependency local launcher.
# Serves the folder this script sits in over HTTP on localhost and opens a browser. Uses only .NET classes that
# ship with Windows PowerShell, so nothing needs installing - no Python, no Node, no admin rights.
# Close this window (or press Ctrl+C) to stop the server.

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$site = Join-Path $root 'site'
if (-not (Test-Path $site)) { $site = $root }

$mime = @{
  '.html' = 'text/html; charset=utf-8'; '.js' = 'text/javascript'; '.mjs' = 'text/javascript'; '.css' = 'text/css'
  '.json' = 'application/json'; '.arrow' = 'application/vnd.apache.arrow.file'; '.svg' = 'image/svg+xml'
  '.png' = 'image/png'; '.jpg' = 'image/jpeg'; '.jpeg' = 'image/jpeg'; '.webp' = 'image/webp'; '.ico' = 'image/x-icon'
  '.woff' = 'font/woff'; '.woff2' = 'font/woff2'; '.ttf' = 'font/ttf'; '.txt' = 'text/plain; charset=utf-8'
  '.csv' = 'text/csv'; '.geojson' = 'application/geo+json'; '.map' = 'application/json'
}

# First free port from 8777 upwards.
$port = 0
foreach ($p in 8777..8797) {
  try {
    $probe = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, $p)
    $probe.Start(); $probe.Stop(); $port = $p; break
  } catch { }
}
if ($port -eq 0) { Write-Host 'No free port between 8777 and 8797.' -ForegroundColor Red; Read-Host 'Press Enter to close'; exit 1 }

$listener = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, $port)
$listener.Start()
$url = "http://localhost:$port/"

Write-Host ''
Write-Host '  Train Link Simulator' -ForegroundColor Cyan
Write-Host "  Serving $site"
Write-Host "  Open: $url" -ForegroundColor Green
Write-Host '  Keep this window open while you use the simulator; close it to stop.'
Write-Host ''
Start-Process $url

function Send-Response {
  param($stream, [int]$code, [string]$status, [byte[]]$body, [string]$type)
  $head = "HTTP/1.1 $code $status`r`nContent-Type: $type`r`nContent-Length: $($body.Length)`r`nCache-Control: no-store`r`nConnection: close`r`n`r`n"
  $headBytes = [System.Text.Encoding]::ASCII.GetBytes($head)
  $stream.Write($headBytes, 0, $headBytes.Length)
  if ($body.Length -gt 0) { $stream.Write($body, 0, $body.Length) }
  $stream.Flush()
}

try {
  while ($true) {
    $client = $listener.AcceptTcpClient()
    try {
      $client.ReceiveTimeout = 5000; $client.SendTimeout = 60000
      $stream = $client.GetStream()
      $reader = [System.IO.StreamReader]::new($stream, [System.Text.Encoding]::ASCII, $false, 1024, $true)
      $request = $reader.ReadLine()
      if (-not $request) { $client.Close(); continue }
      $parts = $request.Split(' ')
      $path = [System.Uri]::UnescapeDataString($parts[1].Split('?')[0])
      if ($path -eq '/') { $path = '/index.html' }
      # Resolve inside the site folder only.
      $target = Join-Path $site ($path.TrimStart('/') -replace '/', '\')
      $full = [System.IO.Path]::GetFullPath($target)
      if (-not $full.StartsWith([System.IO.Path]::GetFullPath($site))) {
        Send-Response $stream 403 'Forbidden' ([byte[]]@()) 'text/plain'
      } elseif (Test-Path $full -PathType Leaf) {
        $bytes = [System.IO.File]::ReadAllBytes($full)
        $ext = [System.IO.Path]::GetExtension($full).ToLower()
        $type = $mime[$ext]; if (-not $type) { $type = 'application/octet-stream' }
        Send-Response $stream 200 'OK' $bytes $type
      } else {
        $msg = [System.Text.Encoding]::UTF8.GetBytes("Not found: $path")
        Send-Response $stream 404 'Not Found' $msg 'text/plain; charset=utf-8'
      }
    } catch {
      # A browser closing a connection mid-transfer is normal; keep serving.
    } finally {
      $client.Close()
    }
  }
} finally {
  $listener.Stop()
}
