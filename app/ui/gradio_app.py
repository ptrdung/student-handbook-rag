import json
import os
import uuid
from pathlib import Path
from typing import Iterator, List, Tuple

import gradio as gr
import httpx
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")
REQUEST_TIMEOUT = float(os.getenv("UI_REQUEST_TIMEOUT", "300"))

ASSETS_DIR = Path(__file__).parent / "assets"
BOT_AVATAR = ASSETS_DIR / "bot.png"
USER_AVATAR = ASSETS_DIR / "user.png"




def _new_session_id() -> str:
    return f"ui-{uuid.uuid4().hex[:12]}"


def _avatar_pair():
    user = str(USER_AVATAR) if USER_AVATAR.exists() else None
    bot = str(BOT_AVATAR) if BOT_AVATAR.exists() else None
    if user is None and bot is None:
        return None
    return (user, bot)


# ---------------- Chat ----------------

def chat_stream(
    message: str,
    history: List[dict],
    session_id: str,
    api_base: str,
) -> Iterator[str]:
    """Stream assistant tokens from the FastAPI /chat/stream endpoint."""
    if not message or not message.strip():
        yield ""
        return

    base = (api_base or API_BASE_URL).rstrip("/")
    url = f"{base}/chat/stream"
    params = {"query": message, "session_id": session_id or "default_session"}

    accumulated = ""
    try:
        with httpx.stream(
            "GET", url, params=params, timeout=REQUEST_TIMEOUT
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = chunk.get("text", "")
                if text:
                    accumulated += text
                    yield accumulated
                if chunk.get("is_last"):
                    break
    except httpx.HTTPError as e:
        yield f"{accumulated}\n\n_⚠️ Lỗi gọi API: {e}_"


def reset_session() -> Tuple[str, str, List]:
    sid = _new_session_id()
    return sid, sid, []


# ---------------- Ingestion ----------------

def upload_document(file, api_base: str) -> str:
    if file is None:
        return "❌ Chưa chọn file."

    file_path = Path(file.name if hasattr(file, "name") else file)
    base = (api_base or API_BASE_URL).rstrip("/")
    url = f"{base}/ingestion/upload"

    try:
        with file_path.open("rb") as fh:
            files = {"file": (file_path.name, fh, "application/pdf")}
            response = httpx.post(url, files=files, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        data = response.json()
        return (
            f"✅ {data.get('message', 'Uploaded.')}\n\n"
            f"**File path:** `{data.get('file_path', '')}`\n\n"
            f"_Pipeline ingest đang chạy nền — bạn có thể chuyển sang tab Chat sau khi hoàn tất._"
        )
    except httpx.HTTPStatusError as e:
        return f"❌ Server trả về {e.response.status_code}: {e.response.text}"
    except httpx.HTTPError as e:
        return f"❌ Lỗi kết nối: {e}"


# ---------------- Evaluation ----------------

def run_evaluation(
    evaluator_name: str,
    data_path: str,
    use_rerank: bool,
    limit_search: int,
    limit_rerank: int,
    api_base: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    base = (api_base or API_BASE_URL).rstrip("/")
    url = f"{base}/evaluation/run"
    payload = {
        "evaluator_name": evaluator_name,
        "data_path": data_path,
        "use_rerank": use_rerank,
        "limit_search": int(limit_search),
        "limit_rerank": int(limit_rerank),
    }

    try:
        response = httpx.post(url, json=payload, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        data = response.json()
    except httpx.HTTPStatusError as e:
        empty = pd.DataFrame()
        return empty, empty, f"❌ Server trả về {e.response.status_code}: {e.response.text}"
    except httpx.HTTPError as e:
        empty = pd.DataFrame()
        return empty, empty, f"❌ Lỗi kết nối: {e}"

    metrics_rows = [
        {"name": m["name"], "value": m["value"]}
        for m in data.get("metrics", [])
    ]
    metrics_df = pd.DataFrame(metrics_rows)

    summary = data.get("summary", {})
    summary_df = pd.DataFrame(
        [{"metric": k, "value": v} for k, v in summary.items()]
    )

    return metrics_df, summary_df, "✅ Hoàn thành đánh giá."


# ---------------- UI ----------------

def build_ui() -> gr.Blocks:
    with gr.Blocks(
        title="Student Handbook RAG",
        fill_height=True,
        fill_width=True,
    ) as demo:
        # Sidebar — config
        with gr.Sidebar(width=300, position="left"):
            gr.Markdown("### ⚙️ Cấu hình")
            session_box = gr.Textbox(
                label="Session ID",
                value=_new_session_id(),
                interactive=False,
            )
            new_session_btn = gr.Button("🆕 Phiên mới", variant="secondary")

            with gr.Accordion("Nâng cao", open=False):
                api_base_box = gr.Textbox(
                    label="API Base URL",
                    value=API_BASE_URL,
                )
                gr.Markdown(
                    "<sub>Đổi nhanh để trỏ tới backend khác mà không cần restart.</sub>"
                )

            gr.Markdown("<sub>Student Handbook RAG · v0.1</sub>")

        gr.Markdown(
            "# 📚 Student Handbook RAG\n"
            "Trợ lý hỏi đáp quy chế, sổ tay sinh viên."
        )

        session_state = gr.State(session_box.value)

        with gr.Tabs():
            # ---- Chat tab ----
            with gr.Tab("💬 Trò chuyện"):
                chatbot = gr.Chatbot(
                    show_label=False,
                    height=560,
                    avatar_images=_avatar_pair(),
                )

                chat = gr.ChatInterface(
                    fn=chat_stream,
                    chatbot=chatbot,
                    additional_inputs=[session_state, api_base_box],
                    examples=[
                        ["Quy định về điểm rèn luyện như thế nào?"],
                        ["Sinh viên bị cảnh báo học tập khi nào?"],
                        ["Thủ tục xin nghỉ học tạm thời?"],
                    ],
                )

            # ---- Ingestion tab ----
            with gr.Tab("📥 Ingest dữ liệu"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown(
                            "**Tải lên tài liệu PDF**\n\n"
                            "Pipeline sẽ chạy nền: extract → chunk → embed → upsert vào vector store."
                        )
                        pdf_input = gr.File(
                            label="Chọn file PDF",
                            file_types=[".pdf"],
                            type="filepath",
                        )
                        upload_btn = gr.Button("Bắt đầu ingest", variant="primary")
                    with gr.Column(scale=1):
                        upload_status = gr.Markdown(
                            value="_Trạng thái sẽ hiện ở đây sau khi upload._"
                        )

                upload_btn.click(
                    fn=upload_document,
                    inputs=[pdf_input, api_base_box],
                    outputs=upload_status,
                )

            # ---- Evaluation tab ----
            with gr.Tab("📊 Đánh giá"):
                gr.Markdown(
                    "Chạy benchmark trên bộ câu hỏi mẫu. Kết quả gồm các metric retrieval (Hit@K, MRR, NDCG)."
                )
                with gr.Row():
                    evaluator_name = gr.Dropdown(
                        choices=["search"],
                        value="search",
                        label="Evaluator",
                    )
                    data_path = gr.Textbox(
                        value="data/evaluation_sample.json",
                        label="Đường dẫn dữ liệu eval",
                    )
                with gr.Row():
                    use_rerank = gr.Checkbox(value=True, label="Use rerank")
                    limit_search = gr.Number(value=20, label="limit_search", precision=0)
                    limit_rerank = gr.Number(value=5, label="limit_rerank", precision=0)

                run_btn = gr.Button("Chạy đánh giá", variant="primary")
                eval_status = gr.Markdown()
                with gr.Row():
                    metrics_table = gr.Dataframe(
                        label="Metrics", interactive=False, wrap=True
                    )
                    summary_table = gr.Dataframe(
                        label="Summary", interactive=False, wrap=True
                    )

                run_btn.click(
                    fn=run_evaluation,
                    inputs=[
                        evaluator_name,
                        data_path,
                        use_rerank,
                        limit_search,
                        limit_rerank,
                        api_base_box,
                    ],
                    outputs=[metrics_table, summary_table, eval_status],
                )

        # Wiring: new session resets state, displayed ID, and clears chat
        new_session_btn.click(
            fn=reset_session,
            outputs=[session_state, session_box, chatbot],
        )

    return demo


def main() -> None:
    demo = build_ui()
    demo.queue().launch(
        server_name=os.getenv("UI_HOST", "0.0.0.0"),
        server_port=int(os.getenv("UI_PORT", "7860")),
        share=os.getenv("UI_SHARE", "false").lower() == "true",
        theme=gr.themes.Soft(),
    )


if __name__ == "__main__":
    main()
