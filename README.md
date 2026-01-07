---
title: DoclingSpace — Markdown/JSON (ZeroGPU)
emoji: 📄
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: '5.0.0'
app_file: app.py
pinned: false
---

# DoclingSpace — Markdown/JSON (ZeroGPU)

Conversor de documentos (formatos suportados pelo Docling) para **Markdown** e/ou **JSON**.

## Upload dir (servidor)

- Os arquivos enviados pelo usuário são copiados para `UPLOAD_DIR`.
- Se o Space tiver **Persistent Storage**, `UPLOAD_DIR` default vira `/data/uploads`.

## ZeroGPU

- Em **Settings → Hardware**, selecione **ZeroGPU**.
- A GPU só é provisionada durante execução de funções decoradas com `@spaces.GPU`.

## Limites para documentos longos

- `max_num_pages` default: **2000**
- `max_file_size_mb` default: **200MB**

Ajuste esses limites na UI conforme necessário.
