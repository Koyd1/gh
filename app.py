from __future__ import annotations

import io
import textwrap
import zipfile
from pathlib import Path
from typing import Dict, List

import streamlit as st

from agents._llm import DEFAULT_MODEL
from graph import CampaignState, prepare_initial_state, run_campaign
from memory import rag
from agents._llm import DEFAULT_MODEL

# import os
# os.environ["LLM_BACKEND"] = "gemini"


st.set_page_config(page_title="AI Orchestrator", layout="wide")
st.title("AI Orchestrator — маркетинговый мини-стартап")

if "latest_state" not in st.session_state:
    st.session_state.latest_state = None
if "run_id" not in st.session_state:
    st.session_state.run_id = None

col_input, col_view = st.columns([1, 2])

with col_input:
    st.header("Бриф")
    default_brief = "SaaS для аналитики продаж в рознице. Цель — 50 лидов в месяц в Румынии и Молдове."
    brief = st.text_area("Опиши продукт, аудиторию и цель", value=default_brief, height=280)

    project_title = st.text_input("Название кампании", value="Retail Analytics SaaS")
    model_name = st.text_input("Модель Ollama", value=DEFAULT_MODEL)

    uploaded_files = st.file_uploader(
        "Дополнительные материалы (Markdown/Text)",
        accept_multiple_files=True,
        type=["txt", "md"],
    )

    extra_documents: Dict[str, str] = {}
    if uploaded_files:
        for file in uploaded_files:
            try:
                content = file.getvalue().decode("utf-8")
            except UnicodeDecodeError:
                st.warning(f"Не удалось прочитать файл {file.name}. Используйте UTF-8.")
                continue
            extra_documents[file.name] = content

    run_button = st.button("Запустить оркестрацию", type="primary")

    if run_button:
        if not brief.strip():
            st.warning("Введите бриф, чтобы запустить процесс.")
        else:
            initial_state = prepare_initial_state(
                brief.strip(),
                model=model_name.strip() or DEFAULT_MODEL,
                project_title=project_title.strip() or "Маркетинговая кампания",
                extra_documents=extra_documents or None,
            )
            st.session_state.run_id = initial_state["run_id"]
            try:
                with st.spinner("Агенты работают над кампанией..."):
                    final_state = run_campaign(initial_state)
            except Exception as exc:  # pragma: no cover - surface errors in UI
                st.error(f"Ошибка при запуске оркестрации: {exc}")
            else:
                st.session_state.latest_state = final_state
                st.success("Готово! Кампания собрана.")

with col_view:
    st.header("Прогресс")
    latest_state: CampaignState | None = st.session_state.get("latest_state")
    board = (latest_state or {}).get("board", {})

    status_columns = {"Backlog": [], "In Progress": [], "Done": []}
    for task_id, task in board.items():
        status = task.get("status", "Backlog")
        if status not in status_columns:
            status_columns[status] = []
        status_columns[status].append((task.get("title", task_id), task.get("owner", ""), task.get("notes", "")))

    backlog_col, doing_col, done_col = st.columns(3)

    def _render_column(container, title: str, items: List[tuple[str, str, str]]) -> None:
        container.subheader(title)
        if not items:
            container.caption("—")
        for name, owner, notes in items:
            container.markdown(f"**{name}**  \
_{owner}_")
            if notes:
                snippet = textwrap.shorten(notes.replace("\n", " "), width=160, placeholder="…")
                container.caption(snippet)

    _render_column(backlog_col, "Backlog", status_columns.get("Backlog", []))
    _render_column(doing_col, "Doing", status_columns.get("In Progress", []))
    _render_column(done_col, "Done", status_columns.get("Done", []))

    st.header("Артефакты")
    run_id = st.session_state.get("run_id")
    artifacts = rag.list_artifacts(run_id) if run_id else []

    for artifact in artifacts:
        with st.expander(artifact.title):
            st.markdown(artifact.path.read_text(encoding="utf-8"))

    if artifacts:
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            for artifact in artifacts:
                archive.write(artifact.path, arcname=artifact.path.name)
        st.download_button(
            "Скачать ZIP",
            data=zip_buffer.getvalue(),
            file_name=f"{run_id}_artifacts.zip",
            mime="application/zip",
        )

    if latest_state and latest_state.get("summary"):
        st.header("Итоговая сводка")
        st.markdown(latest_state["summary"])

    if not latest_state:
        st.info("Запусти процесс, чтобы увидеть результаты.")
