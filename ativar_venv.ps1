# Script para ativar o ambiente virtual no PowerShell

# Verifica se o ambiente virtual já existe
$venvPath = ".venv"

if (-not (Test-Path -Path $venvPath)) {
    Write-Host "Criando ambiente virtual..."
    python -m venv $venvPath
}

# Ativa o ambiente virtual
& "$venvPath\Scripts\Activate.ps1"

# Instala as dependências
Write-Host "Instalando dependências..."
pip install --upgrade pip
pip install -r requirements.txt

Write-Host "Ambiente virtual ativado e dependências instaladas com sucesso!"
