import os
import shutil
import uuid
import asyncio
from asyncio import CancelledError
from pathlib import Path
from pdf2zh import __version__
from pdf2zh.high_level import translate
from pdf2zh.translator import (
    BaseTranslator,
    GoogleTranslator,
    BingTranslator,
    DeepLTranslator,
    DeepLXTranslator,
    OllamaTranslator,
    AzureOpenAITranslator,
    OpenAITranslator,
    ZhipuTranslator,
    ModelScopeTranslator,
    SiliconTranslator,
    GeminiTranslator,
    AzureTranslator,
    TencentTranslator,
    DifyTranslator,
    AnythingLLMTranslator,
)

import gradio as gr
from gradio_pdf import PDF
import tqdm
import requests
import cgi

# 配置部分
service_map = {
    "Google": GoogleTranslator,
    "Bing": BingTranslator,
    "DeepL": DeepLTranslator,
    "DeepLX": DeepLXTranslator,
    "Ollama": OllamaTranslator,
    "AzureOpenAI": AzureOpenAITranslator,
    "OpenAI": OpenAITranslator,
    "Zhipu": ZhipuTranslator,
    "ModelScope": ModelScopeTranslator,
    "Silicon": SiliconTranslator,
    "Gemini": GeminiTranslator,
    "Azure": AzureTranslator,
    "Tencent": TencentTranslator,
    "Dify": DifyTranslator,
    "AnythingLLM": AnythingLLMTranslator,
}

lang_map = {
    "简体中文": "zh",
    "繁体中文": "zh-TW",
    "英语": "en",
    "法语": "fr",
    "德语": "de",
    "日语": "ja",
    "韩语": "ko",
    "俄语": "ru",
    "西班牙语": "es",
    "意大利语": "it",
}

page_map = {
    "全部": None,
    "第一页": [0],
    "前5页": list(range(0, 5)),
}

flag_demo = False
if os.getenv("PDF2ZH_DEMO"):
    flag_demo = True
    service_map = {
        "Google": GoogleTranslator,
    }
    page_map = {
        "第一页": [0],
        "前20页": list(range(0, 20)),
    }
    client_key = os.getenv("PDF2ZH_CLIENT_KEY")
    server_key = os.getenv("PDF2ZH_SERVER_KEY")

# 自定义CSS
custom_css = """
.secondary-text {color: #999 !important;}
footer {visibility: hidden}
.env-warning {color: #dd5500 !important;}
.env-success {color: #559900 !important;}

/* 输入文件区域的样式 */
.input-file {
    border: 1.2px dashed #165DFF !important;
    border-radius: 8px !important;
    padding: 20px;
    background-color: #F9FAFB;
}

/* 按钮样式 */
button.primary {
    background-color: #165DFF !important;
    border-radius: 6px !important;
    padding: 10px 20px !important;
    font-weight: bold;
}

button.secondary {
    background-color: #E0E0E0 !important;
    border-radius: 6px !important;
    padding: 10px 20px !important;
}

/* 进度条样式 */
.progress-bar-wrap {
    border-radius: 8px !important;
    background-color: #E8F3FF !important;
}

.progress-bar {
    border-radius: 8px !important;
    background-color: #165DFF !important;
}
"""

# reCAPTCHA脚本
recaptcha_script = """
<script src="https://www.google.com/recaptcha/api.js?render=explicit" async defer></script>
<script type="text/javascript">
    var onVerify = function(token) {
        el=document.getElementById('verify').getElementsByTagName('textarea')[0];
        el.value=token;
        el.dispatchEvent(new Event('input'));
    };
</script>
"""

cancellation_event_map = {}

# 辅助函数
def verify_recaptcha(response):
    recaptcha_url = "https://www.google.com/recaptcha/api/siteverify"
    print("reCAPTCHA", server_key, response)
    data = {"secret": server_key, "response": response}
    result = requests.post(recaptcha_url, data=data).json()
    print("reCAPTCHA", result.get("success"))
    return result.get("success")

def download_with_limit(url, save_path, size_limit):
    chunk_size = 1024
    total_size = 0
    with requests.get(url, stream=True, timeout=10) as response:
        response.raise_for_status()
        content = response.headers.get("Content-Disposition")
        try:  # 从header获取文件名
            _, params = cgi.parse_header(content)
            filename = params["filename"]
        except Exception:  # 从URL获取文件名
            filename = os.path.basename(url)
        with open(save_path / filename, "wb") as file:
            for chunk in response.iter_content(chunk_size=chunk_size):
                total_size += len(chunk)
                if size_limit and total_size > size_limit:
                    raise gr.Error("文件大小超过限制")
                file.write(chunk)
    return save_path / filename

def stop_translate_file(state):
    session_id = state["session_id"]
    if session_id is None:
        return
    if session_id in cancellation_event_map:
        cancellation_event_map[session_id].set()

def translate_file(
    file_type,
    file_input,
    link_input,
    service,
    lang_from,
    lang_to,
    page_range,
    recaptcha_response,
    state,
    progress=gr.Progress(),
    *envs,
):
    session_id = uuid.uuid4()
    state["session_id"] = session_id
    cancellation_event_map[session_id] = asyncio.Event()
    """使用选定的服务翻译PDF内容。"""
    if flag_demo and not verify_recaptcha(recaptcha_response):
        raise gr.Error("reCAPTCHA验证失败")
    
    progress(0, desc="开始翻译...")
    
    output = Path("pdf2zh_files")
    output.mkdir(parents=True, exist_ok=True)
    
    if file_type == "文件上传":
        if not file_input:
            raise gr.Error("未选择文件")
        file_path = shutil.copy(file_input, output)
    else:
        if not link_input:
            raise gr.Error("未输入链接")
        file_path = download_with_limit(
            link_input,
            output,
            5 * 1024 * 1024 if flag_demo else None,
        )
    
    filename = os.path.splitext(os.path.basename(file_path))[0]
    file_raw = output / f"{filename}.pdf"
    file_mono = output / f"{filename}-mono.pdf"
    file_dual = output / f"{filename}-dual.pdf"
    
    translator = service_map[service]
    selected_page = page_map[page_range]
    lang_from = lang_map[lang_from]
    lang_to = lang_map[lang_to]
    
    _envs = {}
    for i, env in enumerate(translator.envs.items()):
        _envs[env[0]] = envs[i]
    
    print(f"翻译前的文件: {os.listdir(output)}")
    
    def progress_bar(t: tqdm.tqdm):
        progress(t.n / t.total, desc=f"正在翻译第 {t.n} 页，共 {t.total} 页...")
    
    param = {
        "files": [str(file_raw)],
        "pages": selected_page,
        "lang_in": lang_from,
        "lang_out": lang_to,
        "service": f"{translator.name}",
        "output": output,
        "thread": 4,
        "callback": progress_bar,
        "cancellation_event": cancellation_event_map[session_id],
        "envs": _envs,
    }
    try:
        translate(**param)
    except CancelledError:
        del cancellation_event_map[session_id]
        raise gr.Error("翻译已取消")
    print(f"翻译后的文件: {os.listdir(output)}")
    
    if not file_mono.exists() or not file_dual.exists():
        raise gr.Error("未生成翻译结果")
    
    progress(1.0, desc="翻译完成！")
    
    return (
        str(file_mono),
        str(file_mono),
        str(file_dual),
        gr.update(visible=True),
        gr.update(visible=True),
        gr.update(visible=True),
    )

def readuserandpasswd(file_path):
    tuple_list = []
    content = ""
    if len(file_path) == 2:
        try:
            with open(file_path[1], "r", encoding="utf-8") as file:
                content = file.read()
        except FileNotFoundError:
            print(f"错误: 文件 '{file_path[1]}' 未找到。")
    try:
        with open(file_path[0], "r", encoding="utf-8") as file:
            tuple_list = [
                tuple(line.strip().split(",")) for line in file if line.strip()
            ]
    except FileNotFoundError:
        print(f"错误: 文件 '{file_path[0]}' 未找到。")
    return tuple_list, content

def build_ui(service_map, lang_map, page_map, flag_demo, client_key, recaptcha_script):
    with gr.Blocks(
        title="PDFMathTranslate - PDF翻译并保留格式",
        theme=gr.themes.Default(
            primary_hue=custom_blue, spacing_size="md", radius_size="lg"
        ),
        css=custom_css,
        head=recaptcha_script if flag_demo else "",
    ) as demo:
        gr.Markdown("# [Charlii文献翻译宝](https://www.charliiai.com/)")
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## 输入文件")
                file_type = gr.Radio(
                    choices=["文件上传", "链接输入"],
                    label="选择类型",
                    value="文件上传",
                    info="请选择上传本地文件或提供文件链接。"
                )
                file_input = gr.File(
                    label="上传PDF文件",
                    file_count="single",
                    file_types=[".pdf"],
                    type="filepath",
                    elem_classes=["input-file"],
                    visible=True,
                )
                link_input = gr.Textbox(
                    label="输入PDF链接",
                    placeholder="请输入PDF文件的URL",
                    visible=False,
                    interactive=True,
                )
                
                gr.Markdown("## 翻译选项")
                service = gr.Dropdown(
                    label="选择翻译服务",
                    choices=list(service_map.keys()),
                    value="Google",
                    info="请选择用于翻译的服务提供商。"
                )
                envs = []
                for i in range(3):
                    envs.append(
                        gr.Textbox(
                            visible=False,
                            interactive=True,
                        )
                    )
                with gr.Row():
                    lang_from = gr.Dropdown(
                        label="源语言",
                        choices=list(lang_map.keys()),
                        value="英语",
                        info="请选择PDF文件的源语言。"
                    )
                    lang_to = gr.Dropdown(
                        label="目标语言",
                        choices=list(lang_map.keys()),
                        value="简体中文",
                        info="请选择翻译后的目标语言。"
                    )
                page_range = gr.Radio(
                    choices=list(page_map.keys()),
                    label="选择翻译页面范围",
                    value="全部",
                    info="请选择需要翻译的页面范围。"
                )
                
                recaptcha_response = gr.Textbox(
                    label="reCAPTCHA 响应",
                    elem_id="verify",
                    visible=False
                )
                recaptcha_box = gr.HTML('<div id="recaptcha-box"></div>')
                translate_btn = gr.Button("开始翻译", variant="primary")
                cancellation_btn = gr.Button("取消", variant="secondary")
                tech_details_tog = gr.Markdown(
                    f"""
                    <details>
                        <summary>关于 Charlii</summary>
                        **Charlii** 是一个由清华、北大、哈工大等知名学府博士生群体组织，致力于人工智能（AI）与科研领域的前沿研究与应用。通过关注 **Charlii**，您将获取最新的 AI 技术动态、科研干货和实用教程，助您在学术与职业发展中保持领先。
                        
                        - **关注链接**: [点击这里关注 Charlii](https://ylb.charliiai.com/s/a1ZlD)
                        - **加入群组**: [立即加入我们的社区](https://ylb.charliiai.com/s/KS4e2)
                    </details>
                    """,
                    elem_classes=["secondary-text"],
                )
                service.select(
                    on_select_service,
                    service,
                    envs,
                )
                file_type.select(
                    on_select_filetype,
                    file_type,
                    [file_input, link_input],
                    js=(
                        f"""
                        (a,b)=>{{
                            try{{
                                grecaptcha.render('recaptcha-box',{{
                                    'sitekey':'{client_key}',
                                    'callback':'onVerify'
                                }});
                            }}catch(error){{}}
                            return [a];
                        }}
                        """
                        if flag_demo
                        else ""
                    ),
                )
        
            with gr.Column(scale=2):
                gr.Markdown("## 预览")
                preview = PDF(label="文档预览", visible=True)
        
        # 事件处理
        file_input.upload(
            lambda x: x,
            inputs=file_input,
            outputs=preview,
            js=(
                f"""
                (a,b)=>{{
                    try{{
                        grecaptcha.render('recaptcha-box',{{
                            'sitekey':'{client_key}',
                            'callback':'onVerify'
                        }});
                    }}catch(error){{}}
                    return [a];
                }}
                """
                if flag_demo
                else ""
            ),
        )
        
        state = gr.State({"session_id": None})
        
        translate_btn.click(
            translate_file,
            inputs=[
                file_type,
                file_input,
                link_input,
                service,
                lang_from,
                lang_to,
                page_range,
                recaptcha_response,
                state,
                *envs,
            ],
            outputs=[
                output_file_mono,
                preview,
                output_file_dual,
                output_file_mono,
                output_file_dual,
                output_title,
            ],
        ).then(lambda: None, js="()=>{grecaptcha.reset()}" if flag_demo else "")
        
        cancellation_btn.click(
            stop_translate_file,
            inputs=[state],
        )
        
        return demo

def setup_gui(share=False, authfile=["", ""]):
    userlist, html = readuserandpasswd(authfile)
    if flag_demo:
        demo = build_ui(service_map, lang_map, page_map, flag_demo, client_key, recaptcha_script)
        demo.launch(server_name="0.0.0.0", max_file_size="5mb", inbrowser=True)
    else:
        demo = build_ui(service_map, lang_map, page_map, flag_demo, client_key, recaptcha_script)
        if len(userlist) == 0:
            try:
                demo.launch(
                    server_name="0.0.0.0", debug=True, inbrowser=True, share=share
                )
            except Exception:
                print(
                    "使用 0.0.0.0 启动 GUI 时出错。\n这可能是由于代理软件的全局模式导致。"
                )
                try:
                    demo.launch(
                        server_name="127.0.0.1", debug=True, inbrowser=True, share=share
                    )
                except Exception:
                    print(
                        "使用 127.0.0.1 启动 GUI 时出错。\n这可能是由于代理软件的全局模式导致。"
                    )
                    demo.launch(debug=True, inbrowser=True, share=True)
        else:
            try:
                demo.launch(
                    server_name="0.0.0.0",
                    debug=True,
                    inbrowser=True,
                    share=share,
                    auth=userlist,
                    auth_message=html,
                )
            except Exception:
                print(
                    "使用 0.0.0.0 启动 GUI 时出错。\n这可能是由于代理软件的全局模式导致。"
                )
                try:
                    demo.launch(
                        server_name="127.0.0.1",
                        debug=True,
                        inbrowser=True,
                        share=share,
                        auth=userlist,
                        auth_message=html,
                    )
                except Exception:
                    print(
                        "使用 127.0.0.1 启动 GUI 时出错。\n这可能是由于代理软件的全局模式导致。"
                    )
                    demo.launch(
                        debug=True,
                        inbrowser=True,
                        share=True,
                        auth=userlist,
                        auth_message=html,
                    )

# For auto-reloading while developing
if __name__ == "__main__":
    setup_gui()
