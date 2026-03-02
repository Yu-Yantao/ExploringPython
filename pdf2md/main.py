"""
marker-pdf 1.10.2 Demo with Ollama LLM
使用 Ollama 的 qwen3:14b 模型进行 PDF 转 Markdown
"""

import json
import logging
import os
from pathlib import Path

from marker.config.parser import ConfigParser
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered


# 配置日志级别
def setup_logging(level=logging.DEBUG):
    """
    设置日志级别

    Args:
        level: 日志级别，可选值：
               - logging.DEBUG: 详细的调试信息
               - logging.INFO: 一般信息（默认）
               - logging.WARNING: 警告信息
               - logging.ERROR: 错误信息
               - logging.CRITICAL: 严重错误
    """
    logging.basicConfig(level=level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    # 设置 marker 相关库的日志级别
    logging.getLogger("marker").setLevel(level)



def convert_pdf_with_llm(
    pdf_path: str,
    output_dir: str = "output",
    ollama_url: str = "http://192.168.90.213:11434",
    model_name: str = "qwen3:14b",
    log_level: int = logging.DEBUG,
):
    """
    使用 marker-pdf 1.10.2 和 Ollama LLM 转换 PDF 为 Markdown

    Args:
        pdf_path: PDF 文件路径
        output_dir: 输出目录
        ollama_url: Ollama 服务地址
        model_name: Ollama 模型名称
        log_level: 日志级别（logging.DEBUG/INFO/WARNING/ERROR）
    """
    # 设置日志
    setup_logging(log_level)
    # 确保输出目录存在
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 设置模型缓存路径 - marker-pdf 专用
    os.environ["MARKER_CACHE_DIR"] = r"D:\models\marker"

    print(f"正在转换 PDF: {pdf_path}")
    print(f"使用 Ollama 模型: {model_name} @ {ollama_url}")
    print("正在加载模型...")

    # 配置参数
    config = {
        "output_format": "markdown",  # 输出格式：markdown, json, html, chunks
        "use_llm": True,  # 启用 LLM 模式
        "llm_service": "marker.services.ollama.OllamaService",  # 使用 Ollama 服务
        "ollama_model": model_name,  # 指定模型
        "ollama_base_url": ollama_url,
        "force_ocr": False,  # 是否强制 OCR
        "paginate_output": False,  # 是否分页输出
        "disable_image_extraction": False,  # 是否禁用图片提取
    }

    # 创建配置解析器
    config_parser = ConfigParser(config)

    # 创建转换器
    converter = PdfConverter(
        config=config_parser.generate_config_dict(),
        artifact_dict=create_model_dict(),
        processor_list=config_parser.get_processors(),
        renderer=config_parser.get_renderer(),
        llm_service="marker.services.ollama.OllamaService",
    )

    print("开始转换...")

    # 执行转换
    rendered = converter(pdf_path)

    # 提取文本和图片
    full_text, _, images = text_from_rendered(rendered)

    # 生成输出文件名
    pdf_name = Path(pdf_path).stem
    output_md = output_path / f"{pdf_name}.md"
    output_meta = output_path / f"{pdf_name}_meta.json"

    # 保存 Markdown 文件
    with open(output_md, "w", encoding="utf-8") as f:
        f.write(full_text)

    # 保存元数据
    metadata = rendered.metadata if hasattr(rendered, "metadata") else {}
    with open(output_meta, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    # 保存图片（如果有）
    if images:
        images_dir = output_path / f"{pdf_name}_images"
        images_dir.mkdir(exist_ok=True)
        for img_name, img_data in images.items():
            img_path = images_dir / img_name
            with open(img_path, "wb") as f:
                f.write(img_data)
        print(f"保存了 {len(images)} 张图片到: {images_dir}")

    print(f"\n✅ 转换完成！")
    print(f"📄 Markdown 文件: {output_md}")
    print(f"📊 元数据文件: {output_meta}")

    return full_text, images, metadata


if __name__ == "__main__":
    # 设置全局日志级别（可选：logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR）
    setup_logging(logging.DEBUG)  # 改为 logging.DEBUG 可以看到更详细的信息

    # 示例用法
    pdf_file = "test.pdf"  # 替换为你的 PDF 文件路径

    # 检查文件是否存在
    if not Path(pdf_file).exists():
        print(f"❌ 错误: 找不到文件 {pdf_file}")
        print(f"当前工作目录: {os.getcwd()}")
        print("请将 PDF 文件放在当前目录，或修改 pdf_file 变量为完整路径")
    else:
        # 转换 PDF
        convert_pdf_with_llm(
            pdf_path=pdf_file,
            output_dir="output",
            ollama_url="http://192.168.90.213:11434",
            model_name="qwen3:14b",
            log_level=logging.INFO,  # 可以单独设置这次转换的日志级别
        )
