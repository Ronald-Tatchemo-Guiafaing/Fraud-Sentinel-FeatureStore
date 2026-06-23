# Publier Fraud Sentinel sur GitHub (Ronald-Tatchemo-Guiafaing)
# Exécuter dans PowerShell depuis ce dossier.

$ErrorActionPreference = "Stop"
$env:Path = "C:\Program Files\Git\cmd;C:\Program Files\GitHub CLI;" + $env:Path
Set-Location $PSScriptRoot

Write-Host "1) Connexion GitHub (si demandé, valider dans le navigateur)..."
gh auth status 2>$null
if ($LASTEXITCODE -ne 0) {
    gh auth login --hostname github.com --git-protocol https --web
}

Write-Host "2) Création du dépôt public + push..."
gh repo create Fraud-Sentinel-FeatureStore --public --source=. --remote=origin --push --description "Fraud detection feature store — 3 CNP datasets (ULB, IEEE-CIS, Sparkov)"

Write-Host "Terminé : https://github.com/Ronald-Tatchemo-Guiafaing/Fraud-Sentinel-FeatureStore"
