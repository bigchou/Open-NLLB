import os
import subprocess
import tempfile
import re
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any

app = FastAPI(title="NLLB Evaluation API")

# 固定路徑設定
REFERENCE_DIR = "/alghome/timmy.wan/translation/Open-NLLB/flores200_dataset"
METRIC = "spbleu_flores200"
TEMP_BASE_PATH = "/tmp/nllb_eval_api"
os.makedirs(TEMP_BASE_PATH, exist_ok=True)


class EvalRequest(BaseModel):
    corpus: str
    translate_dir: str
    pivot: str


def parse_metrics_to_dict(raw_stdout: str) -> Dict[str, Any]:
    """
    將 NLLB 輸出的文字表格解析為結構化字典
    """
    # 1. 尋找 "Metric ::" 開始的區塊
    match = re.search(r"Metric :: (.*)\n([\s\S]*)", raw_stdout)
    if not match:
        return {"raw_output": raw_stdout}

    metric_name = match.group(1).strip()
    content = match.group(2).strip()

    # 2. 以分隔線拆分：[0] 是 Summary, [1] 是 Details 表格
    parts = re.split(r"-{10,}", content)

    # 解析 Summary (例如: jpn_Jpan-xx 24.33)
    summary = {}
    summary_lines = parts[0].strip().split('\n')
    for line in summary_lines:
        cols = line.split()
        if len(cols) == 2:
            try:
                summary[cols[0]] = float(cols[1])
            except ValueError:
                continue

    # 解析 Details 表格
    details = []
    if len(parts) > 1:
        table_lines = [l.strip() for l in parts[1].strip().split('\n') if l.strip()]
        if len(table_lines) > 1:
            # 第一行是 Header: ['langs', 'jpn_Jpan-xx', 'xx-jpn_Jpan']
            headers = table_lines[0].split()

            # 之後每一行是數據: ['tgl_Latn', '19.2', '11.0']
            for line in table_lines[1:]:
                cols = line.split()
                if len(cols) == len(headers):
                    entry = {"tgt_lang": cols[0]}
                    for i in range(1, len(headers)):
                        try:
                            entry[headers[i]] = float(cols[i])
                        except ValueError:
                            entry[headers[i]] = cols[i]
                    details.append(entry)

    return {
        "metric": metric_name,
        "summary": summary,
        "details": details
    }

@app.post("/evaluate")
async def run_evaluation(req: EvalRequest):
    # 建立唯一的暫存資料夾
    prefix_str = f"eval_{req.pivot}_{req.corpus}_"
    target_output_dir = tempfile.mkdtemp(prefix=prefix_str, dir=TEMP_BASE_PATH)

    env = os.environ.copy()
    env["PYTHONPATH"] = "."

    command = [
        "python", "examples/nllb/evaluation/calculate_metrics.py",
        "--corpus", req.corpus,
        "--translate-dir", req.translate_dir,
        "--reference-dir", REFERENCE_DIR,
        "--metric", METRIC,
        "--output-dir", target_output_dir,
        "--pivot", req.pivot
    ]

    try:
        # 執行評估程式
        process = subprocess.run(command, env=env, capture_output=True, text=True, check=True)

        # 解析結果
        structured_data = parse_metrics_to_dict(process.stdout)

        return {
            "status": "success",
            "metadata": {
                "corpus": req.corpus,
                "pivot": req.pivot,
                "output_dir": target_output_dir
            },
            "data": structured_data
        }

    except subprocess.CalledProcessError as e:
        # 回傳錯誤訊息
        return {
            "status": "error",
            "message": "Subprocess failed",
            "stderr": e.stderr
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8787)
