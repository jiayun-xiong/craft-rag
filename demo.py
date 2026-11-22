"""Launch the measured classroom configuration without editing .env."""
import os
os.environ.setdefault("USTC_MODEL", "qwen3.8-chat")
from app import PDFS, create_app

if __name__ == "__main__":
    create_app().launch(server_name="127.0.0.1",server_port=7861,allowed_paths=[str(PDFS)],share=False)
