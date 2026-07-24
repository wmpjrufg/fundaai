---
tags: [issue, alto]
file: requirements.txt
severity: alto
---

# Issue — `requirements.txt` em UTF-16/BOM

## Sintoma

A leitura do arquivo apresenta caracteres com espaçamento bizarro:

```
��f q d n = = 1 . 5 . 1
 i s o d u r a t i o n = = 2 0 . 1 1 . 0
 ...
```

Isso é o sinal clássico de um arquivo salvo em **UTF-16 LE com BOM** sendo lido como UTF-8.

## Por que é problema

`pip install -r requirements.txt` falha ou interpreta cada caractere como um pacote inválido. O `env-setup.py` ([[04_Codigo/env-setup.py]]) depende deste arquivo para bootstrap.

## Diagnóstico

`pip-chill > requirements.txt` em PowerShell salva por padrão em UTF-16. Provavelmente foi gerado no Windows e commitado sem conversão.

## Correção sugerida (a confirmar)

Re-salvar em **UTF-8** (sem BOM). Por exemplo, no PowerShell:

```powershell
Get-Content requirements.txt -Encoding Unicode | Set-Content requirements.utf8.txt -Encoding UTF8
```

Ou simplesmente reabrir num editor decente e salvar como UTF-8.

Pacotes esperados (parseados):
```
fqdn==1.5.1
isoduration==20.11.0
jsonpointer==3.0.0
jupyter==1.1.1
mealpy==3.0.3
openpyxl==3.1.5
pip-chill==1.0.3
rfc3987-syntax==1.1.0
scikit-learn==1.7.2
streamlit==1.52.1
tinycss2==1.4.0
uri-template==1.3.0
webcolors==25.10.0
xlsxwriter==3.2.9
ezdxf==1.4.3
```

⚠️ Faltam imports usados pelo código que **não estão** no requirements: `pandas`, `numpy`, `scipy`, `matplotlib`, `joblib`, `playwright` (este último está em `ops/requirements.txt`?). Confirmar.

## Vínculo

- [[01_Projeto/Stack Tecnológico]]
- [[04_Codigo/env-setup.py]]
- [[07_Issues/Lista Mestre de Issues]]
