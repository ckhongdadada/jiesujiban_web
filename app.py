"""Compatibility entry point for the 接诉即办 Flask application."""

from __future__ import annotations

import torch

from src.jsjb.core.config import load_runtime_config
from src.jsjb.reply_generation.service import generator_status
from src.jsjb.web.app import create_app

app = create_app()


if __name__ == "__main__":
    print("=" * 55)
    print("  接诉即办智能服务系统")
    print("=" * 55)

    config = load_runtime_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  运行设备: {device}")

    print("  正在检查模型产物...")
    gen_state = generator_status(
        config.generator_base_model,
        config.generator_lora_dir,
        config.generator_draft_model,
        enable_assisted_decoding=config.enable_assisted_decoding,
    )
    if gen_state["runtime_ready"]:
        print("  生成模型产物已就绪 ✓")
    else:
        print(f"  生成模型未就绪: {', '.join(gen_state['missing']) or '缺少必要文件'}")

    print(f"  访问地址: http://127.0.0.1:{config.enhanced_port}")
    print("=" * 55)

    app.run(host="0.0.0.0", port=config.enhanced_port, debug=False)
