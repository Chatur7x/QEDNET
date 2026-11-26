param([switch]$ApplySchema)

$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$envFile = Join-Path $root "aiven.local.env"
if (-not (Test-Path -LiteralPath $envFile)) {
    throw "Create aiven.local.env from aiven.local.env.example and enter your Aiven connection URI locally. Do not commit or paste it into chat."
}

$databaseUrl = (Get-Content -LiteralPath $envFile | Where-Object { $_ -match '^\s*DATABASE_URL\s*=' } | Select-Object -First 1) -replace '^\s*DATABASE_URL\s*=\s*', ''
$databaseUrl = $databaseUrl.Trim().Trim([char[]]@('"', "'"))
if (-not $databaseUrl -or $databaseUrl -match 'URL_ENCODED_PASSWORD|AIVEN_HOST|AIVEN_PORT') {
    throw "DATABASE_URL in aiven.local.env still contains placeholders."
}
if ($databaseUrl -notmatch '^postgresql\+psycopg://.*[?&]sslmode=require(?:&|$)') {
    throw "Use a postgresql+psycopg URI with sslmode=require. The URI value is not displayed."
}

$oldDatabaseUrl = $env:DATABASE_URL
$oldPythonPath = $env:PYTHONPATH
$env:DATABASE_URL = $databaseUrl
$env:PYTHONPATH = Join-Path $root "backend"
$python = if (Test-Path (Join-Path $root ".venv\Scripts\python.exe")) {
    Join-Path $root ".venv\Scripts\python.exe"
} else {
    "python"
}

Push-Location $root
try {
    & $python -c 'import os; from sqlalchemy import create_engine; engine=create_engine(os.environ["DATABASE_URL"], connect_args={"connect_timeout":8}); connection=engine.connect(); print("Aiven PostgreSQL connectivity: OK; SELECT 1:", connection.exec_driver_sql("SELECT 1").scalar_one()); connection.close(); engine.dispose()'
    if ($LASTEXITCODE -ne 0) { throw "Database connectivity check failed; see the error above. The connection URI was not printed by this script." }

    if ($ApplySchema) {
        Write-Warning "This applies QEDNET's Alembic schema to the configured database. Confirm it is a dedicated, empty QEDNET database before continuing."
        Push-Location (Join-Path $root "backend")
        try {
            & $python -m alembic -c alembic.ini upgrade head
            if ($LASTEXITCODE -ne 0) { throw "Alembic migration failed." }
        } finally {
            Pop-Location
        }
        Write-Output "QEDNET schema migration: applied."
    }
} finally {
    if ($null -eq $oldDatabaseUrl) { Remove-Item Env:DATABASE_URL -ErrorAction SilentlyContinue } else { $env:DATABASE_URL = $oldDatabaseUrl }
    if ($null -eq $oldPythonPath) { Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue } else { $env:PYTHONPATH = $oldPythonPath }
    Pop-Location
}
