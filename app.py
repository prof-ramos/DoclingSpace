import json
import os
import re
import shutil
import tempfile
import zipfile
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import List, Tuple

import gradio as gr
import pandas as pd
import spaces
import torch

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions


# ---------------------------
# Config (uploads / storage)
# ---------------------------


def _default_upload_dir() -> Path:
    # Se houver Persistent Storage, o mount padrão é /data
    if Path("/data").exists() and os.access("/data", os.W_OK):
        return Path("/data/uploads")
    return Path(tempfile.gettempdir()) / "uploads"


UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", str(_default_upload_dir())))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def _safe_stem(name: str) -> str:
    stem = Path(name).stem
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("_")
    return stem or "document"


def _unique_name(original_name: str) -> str:
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
    return f"{_safe_stem(original_name)}_{ts}{Path(original_name).suffix}"


def _persist_uploads(filepaths: List[str]) -> Tuple[List[str], List[dict]]:
    """Copia arquivos enviados para UPLOAD_DIR e retorna os paths persistidos + tabela."""
    persisted = []
    rows = []
    for fp in filepaths:
        src = Path(fp)
        if not src.exists():
            rows.append(
                {
                    "arquivo": src.name,
                    "upload_path": "",
                    "status": "ERRO",
                    "detalhe": "arquivo não encontrado",
                }
            )
            continue
        dst = UPLOAD_DIR / _unique_name(src.name)
        shutil.copy2(src, dst)
        persisted.append(str(dst))
        rows.append(
            {
                "arquivo": src.name,
                "upload_path": str(dst),
                "status": "OK",
                "detalhe": "",
            }
        )
    return persisted, rows


# ---------------------------
# Docling converter
# ---------------------------


def _gpu_duration(
    files, outputs, ocr_for_pdf: bool, max_num_pages: int, max_file_size_mb: int
) -> int:
    """Duração dinâmica para ZeroGPU.

    *Sem leitura prévia de páginas*: usa max_num_pages como proxy de custo.
    """
    try:
        n = len(files) if isinstance(files, list) else 1
    except Exception:
        n = 1

    base = 120 + 45 * max(0, n - 1)

    # proxies de custo
    if int(max_num_pages) > 500:
        base += 240
    if int(max_num_pages) > 1000:
        base += 240
    if bool(ocr_for_pdf):
        base += 180

    # teto conservador (ajuste se necessário)
    return max(120, min(base, 900))


@lru_cache(maxsize=4)
def _cached_converter(ocr_for_pdf: bool) -> DocumentConverter:
    pdf_opts = PdfPipelineOptions()
    pdf_opts.do_ocr = bool(ocr_for_pdf)

    artifacts_path = os.getenv("DOCLING_ARTIFACTS_PATH")
    if artifacts_path:
        pdf_opts.artifacts_path = artifacts_path

    return DocumentConverter(
        # Se você quiser whitelisting, ajuste allowed_formats.
        # Para "todos os formatos suportados", omita allowed_formats.
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_opts)}
    )


def _normalize_filepaths(files) -> List[str]:
    filepaths: List[str] = []
    for f in files or []:
        if isinstance(f, str):
            filepaths.append(f)
        elif isinstance(f, dict) and "path" in f:
            filepaths.append(f["path"])
        elif hasattr(f, "name"):
            filepaths.append(f.name)
    return filepaths


# ---------------------------
# Main conversion (ZeroGPU)
# ---------------------------


@spaces.GPU(duration=_gpu_duration)
def convert_files(
    files,
    outputs: List[str],
    ocr_for_pdf: bool,
    max_num_pages: int,
    max_file_size_mb: int,
):
    filepaths = _normalize_filepaths(files)
    if not filepaths:
        raise gr.Error("Envie pelo menos 1 arquivo.")

    want_md = "md" in outputs
    want_json = "json" in outputs
    if not (want_md or want_json):
        raise gr.Error("Selecione pelo menos um formato de saída (Markdown e/ou JSON).")

    max_num_pages = int(max_num_pages)
    max_file_size_mb = int(max_file_size_mb)

    # Persiste uploads em UPLOAD_DIR
    persisted_paths, upload_rows = _persist_uploads(filepaths)

    # Sanity check: GPU deve aparecer apenas dentro do decorator
    gpu_note = ""
    try:
        t = torch.tensor([0.0]).cuda()
        gpu_note = f"GPU ok: {t.device}"
    except Exception as e:
        gpu_note = f"GPU indisponível (ok em CPU): {type(e).__name__}: {e}"

    converter = _cached_converter(bool(ocr_for_pdf))

    out_dir = Path(tempfile.mkdtemp(prefix="docling_out_"))

    rows = []
    md_preview = ""
    json_preview = None

    # Batch: não aborta no primeiro erro
    conv_iter = converter.convert_all(
        persisted_paths,
        raises_on_error=False,
        max_num_pages=max_num_pages,
        max_file_size=max_file_size_mb * 1024 * 1024,
    )

    for i, (src_path, res) in enumerate(zip(persisted_paths, conv_iter)):
        src = Path(src_path)
        base = _safe_stem(src.name)

        try:
            doc = getattr(res, "document", None)
            if doc is None:
                raise RuntimeError("Conversão falhou: documento não gerado")

            produced = []

            if want_md:
                md = doc.export_to_markdown()
                md_path = out_dir / f"{base}.md"
                md_path.write_text(md, encoding="utf-8")
                produced.append(md_path.name)
                if i == 0:
                    md_preview = md

            if want_json:
                d = doc.export_to_dict()
                json_path = out_dir / f"{base}.json"
                json_path.write_text(
                    json.dumps(d, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                produced.append(json_path.name)
                if i == 0:
                    json_preview = d

            rows.append(
                {
                    "arquivo": src.name,
                    "status": "OK",
                    "saidas": ", ".join(produced) if produced else "(nenhuma)",
                    "erro": "",
                }
            )

        except Exception as e:
            rows.append(
                {
                    "arquivo": src.name,
                    "status": "ERRO",
                    "saidas": "",
                    "erro": f"{type(e).__name__}: {e}",
                }
            )

    # Empacota resultados em .zip
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    zip_path = out_dir / f"docling_outputs_{ts}.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in out_dir.iterdir():
            if p.is_file() and p.name != zip_path.name:
                z.write(p, arcname=p.name)

    df = pd.DataFrame(rows)
    df_uploads = pd.DataFrame(upload_rows)

    # Trunca preview para não travar UI
    if md_preview and len(md_preview) > 120_000:
        md_preview = md_preview[:120_000] + "\n\n<!-- preview truncado -->\n"

    return (
        str(zip_path),
        df,
        df_uploads,
        md_preview,
        json_preview,
        gpu_note,
        str(UPLOAD_DIR),
    )


# ---------------------------
# UI
# ---------------------------

with gr.Blocks(title="DoclingSpace — Markdown/JSON (ZeroGPU)") as demo:
    gr.Markdown(
        """
# DoclingSpace — Markdown/JSON (ZeroGPU)

1) Faça upload de 1 ou mais arquivos.
2) Ajuste limites para documentos longos (ex.: **2000 páginas**).
3) Converta e baixe o `.zip`.
"""
    )

    upload_dir_box = gr.Textbox(
        value=str(UPLOAD_DIR),
        label="Diretório de upload no servidor (UPLOAD_DIR)",
        interactive=False,
    )

    files = gr.Files(
        label="Upload de documentos", file_count="multiple", type="filepath"
    )

    with gr.Row():
        outputs = gr.CheckboxGroup(
            choices=["md", "json"],
            value=["md"],
            label="Saídas",
        )
        ocr_for_pdf = gr.Checkbox(
            value=False,
            label="OCR para PDFs (útil para scans; mais lento)",
        )

    with gr.Accordion("Limites (recomendado para documentos longos)", open=True):
        max_num_pages = gr.Slider(
            minimum=1,
            maximum=5000,
            value=2000,
            step=1,
            label="Máximo de páginas por documento (PDF)",
        )
        max_file_size_mb = gr.Slider(
            minimum=1,
            maximum=1000,
            value=200,
            step=1,
            label="Tamanho máximo por arquivo (MB)",
        )

    run_btn = gr.Button("Converter", variant="primary")

    zip_out = gr.File(label="Baixar resultados (.zip)")

    status_df = gr.Dataframe(
        label="Status de conversão",
        interactive=False,
        wrap=True,
        datatype=["str", "str", "str", "str"],
        headers=["arquivo", "status", "saidas", "erro"],
    )

    uploads_df = gr.Dataframe(
        label="Uploads persistidos (servidor)",
        interactive=False,
        wrap=True,
        datatype=["str", "str", "str", "str"],
        headers=["arquivo", "upload_path", "status", "detalhe"],
    )

    with gr.Accordion("Prévia (primeiro arquivo)", open=False):
        md_prev = gr.Markdown(label="Markdown")
        json_prev = gr.JSON(label="JSON")

    gpu_status = gr.Textbox(label="Status GPU (sanity check)", interactive=False)

    run_btn.click(
        fn=convert_files,
        inputs=[files, outputs, ocr_for_pdf, max_num_pages, max_file_size_mb],
        outputs=[
            zip_out,
            status_df,
            uploads_df,
            md_prev,
            json_prev,
            gpu_status,
            upload_dir_box,
        ],
        concurrency_limit=1,
        concurrency_id="gpu",
    )

    # Jobs longos: limite fila e concorrência
    demo.queue(max_size=2, default_concurrency_limit=1)


if __name__ == "__main__":
    demo.launch()
