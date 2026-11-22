"""Local Gradio application for machining evidence and teaching-cell NC output."""
import html
from pathlib import Path
from urllib.parse import quote
import gradio as gr
from src.generation.pipeline import ask, answer_text
from src.nc.turning import Setup
from nc_job import export
from src.retrieval.store import ROOT

PDFS = ROOT / "data/raw_pdfs"
EXAMPLES = ROOT / "examples"


def respond(question, history, messages):
    try:
        result = ask(question, history)
    except Exception as exc:
        raise gr.Error(f"问答失败，请检查知识库和服务配置（{type(exc).__name__}）") from None
    text = answer_text(result)
    sources = []
    for key, ref in result["citations"].items():
        name = Path(ref["source"]).name
        page = int(ref["page"])
        url = "/gradio_api/file=" + quote(str(PDFS/name)) + f"#page={page}"
        sources.append(f'<details><summary>[{key}] {html.escape(name)} · 第 {page} 页</summary>'
                       f'<a href="{url}" target="_blank">打开 PDF 原页</a><pre>{html.escape(ref["text"])}</pre></details>')
    turns = (history + [{"问题":result["standalone"], "答案":text}])[-3:]
    messages = list(messages or []) + [{"role":"user","content":question},{"role":"assistant","content":text}]
    return "", turns, messages, "".join(sources), result


def generate(dxf, config_text):
    try:
        path = export(Path(dxf) if dxf else EXAMPLES/"两阶外圆.dxf", Setup.model_validate_json(config_text),ROOT/"output/jobs")
    except Exception as exc:
        raise gr.Error(str(exc) if isinstance(exc,ValueError) else type(exc).__name__) from None
    return str(path/"O0701.nc"),str(path/"工序卡.json")


def create_app():
    with gr.Blocks(title="车削工艺助手",analytics_enabled=False) as ui:
        gr.Markdown("# 车削工艺助手")
        with gr.Tab("工艺问答"):
            state=gr.State([])
            chat=gr.Chatbot(label="工艺记录")
            q=gr.Textbox(label="零件要求或追问",placeholder="传动轴粗车和精车分别用哪台机床？",max_lines=4)
            with gr.Row():
                submit=gr.Button("查询",variant="primary")
                clear=gr.Button("新会话")
            evidence=gr.HTML(label="最新回答的原文依据")
            details=gr.JSON(label="结构化结果",open=False)
            gr.on(triggers=[q.submit,submit.click],fn=respond,inputs=[q,state,chat],outputs=[q,state,chat,evidence,details],api_name="ask")
            clear.click(lambda: ("",[],[],"",None),outputs=[q,state,chat,evidence,details])
        with gr.Tab("外圆 NC 工单"):
            gr.Markdown("Haas 教学配置 · 毫米、直径编程、G54 端面零点、T0101。留空使用两阶外圆样件。仅验证理想刀尖二维轨迹，实际装夹与控制器需另行复核。")
            drawing=gr.File(label="毫米 DXF，TURN_PROFILE 开放轮廓",file_types=[".dxf"])
            setup=gr.Textbox(label="工单配置",value=(EXAMPLES/"教学工单.json").read_text(),lines=5)
            button=gr.Button("生成并校验程序")
            code=gr.File(label="数控程序")
            card=gr.File(label="工序与原文依据")
            button.click(generate,[drawing,setup],[code,card])
    return ui


if __name__=="__main__":
    create_app().launch(server_name="127.0.0.1",allowed_paths=[str(PDFS)],share=False)
