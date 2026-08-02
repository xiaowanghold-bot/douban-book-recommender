"""验证 Streamlit Cloud 风格的模块搜索路径。"""

import os
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).parent.parent


def test_app_imports_without_project_root_on_sys_path(tmp_path):
    """Cloud 从 app/main.py 启动时，app/ 与 src/ 顶层导入必须可用。"""
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(PROJECT_ROOT / "app"), str(PROJECT_ROOT / "src")]
    )
    result = subprocess.run(
        [sys.executable, "-c", "import data_loader; import coldstart_predictor"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, result.stderr
