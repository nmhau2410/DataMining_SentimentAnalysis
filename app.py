import json
import os
import sys
import unicodedata
import re
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
import joblib
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

BASE_DIR = Path(__file__).resolve().parent
WEB_DIR = BASE_DIR / "web"
CHECKPOINTS_DIR = BASE_DIR / "checkpoints"

class TextCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_filters, kernel_sizes, dropout=0.5, num_classes=3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(embedding_dim, num_filters, kernel_size=k)
            for k in kernel_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)
    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(0, 2, 1)
        conv_outs = []
        for conv in self.convs:
            out = torch.relu(conv(x))
            out = torch.max(out, dim=2).values
            conv_outs.append(out)
        x = torch.cat(conv_outs, dim=1)
        x = self.dropout(x)
        return self.fc(x)
loaded_models = {}
LABEL_NAMES = {
    0: ("Tiêu cực (Negative)", "badge-neg"),
    1: ("Trung tính (Neutral)", "badge-neu"),
    2: ("Tích cực (Positive)", "badge-pos")
}
def load_all_checkpoints():
    global loaded_models
    nb_path = CHECKPOINTS_DIR / "multinomial_nb_final.joblib"
    if nb_path.exists():
        loaded_models["Multinomial Naive Bayes"] = joblib.load(nb_path)
    lr_path = CHECKPOINTS_DIR / "logistic_regression_final.joblib"
    if lr_path.exists():
        loaded_models["Logistic Regression"] = joblib.load(lr_path)
    svm_path = CHECKPOINTS_DIR / "linear_svm_final.joblib"
    if svm_path.exists():
        loaded_models["Support Vector Machine (SVM)"] = joblib.load(svm_path)
    cnn_path = CHECKPOINTS_DIR / "textcnn" / "textcnn_final_weighted.pt"
    if not cnn_path.exists():
        alt_cnn = CHECKPOINTS_DIR / "textcnn_final_weighted.pt"
        if alt_cnn.exists():
            cnn_path = alt_cnn
    if cnn_path.exists():
        try:
            ckpt = torch.load(str(cnn_path), map_location="cpu", weights_only=False)
            p = ckpt["params"]
            vocab = ckpt["vocab"]
            max_len = ckpt.get("max_length", 64)
            cnn_model = TextCNN(
                vocab_size=len(vocab),
                embedding_dim=p["embedding_dim"],
                num_filters=p["num_filters"],
                kernel_sizes=p["kernel_sizes"],
                dropout=p.get("dropout", 0.5)
            )
            cnn_model.load_state_dict(ckpt["model_state_dict"])
            cnn_model.eval()
            loaded_models["Convolutional Neural Network (TextCNN)"] = {
                "model": cnn_model,
                "vocab": vocab,
                "max_length": max_len
            }
        except Exception as e:
            print(f"Không thể nạp TextCNN: {e}")
    phobert_path = CHECKPOINTS_DIR / "phobert_resume" / "best_epoch.pt"
    if phobert_path.exists() and phobert_path.stat().st_size > 1000000:
        try:
            tok = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
            cfg = AutoConfig.from_pretrained("vinai/phobert-base-v2", num_labels=3)
            phobert_model = AutoModelForSequenceClassification.from_config(cfg)
            sd = torch.load(str(phobert_path), map_location="cpu", weights_only=False)
            phobert_model.load_state_dict(sd)
            phobert_model.eval()
            loaded_models["Transformer (PhoBERT)"] = {
                "model": phobert_model,
                "tokenizer": tok
            }
        except Exception as e:
            print(f"Không thể nạp PhoBERT: {e}")
load_all_checkpoints()
def normalize_vietnamese_text(text: str) -> str:
    """Chuẩn hóa cơ bản ký tự tiếng Việt"""
    text = unicodedata.normalize("NFC", str(text))
    text = re.sub(r"\s+", " ", text).strip()
    return text
def predict_sentiment(text: str, algorithm: str):
    """Thực hiện dự đoán trực tiếp bằng mô hình tương ứng"""
    text_clean = normalize_vietnamese_text(text)
    if algorithm in ["Multinomial Naive Bayes", "Logistic Regression", "Support Vector Machine (SVM)"]:
        model = loaded_models.get(algorithm)
        if model is None:
            raise ValueError(f"Mô hình '{algorithm}' chưa được nạp.")
        pred_id = int(model.predict([text_clean])[0])
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba([text_clean])[0]
        elif hasattr(model, "decision_function"):
            df = model.decision_function([text_clean])[0]
            exp = np.exp(df - np.max(df))
            probs = exp / np.sum(exp)
        else:
            probs = [0.0, 0.0, 0.0]
            probs[pred_id] = 1.0
        p_neg, p_neu, p_pos = float(probs[0]), float(probs[1]), float(probs[2])
        conf = float(probs[pred_id])
    elif algorithm == "Convolutional Neural Network (TextCNN)":
        cnn_data = loaded_models.get(algorithm)
        if cnn_data is None:
            raise ValueError("Mô hình TextCNN chưa được nạp.")
        model = cnn_data["model"]
        vocab = cnn_data["vocab"]
        max_len = cnn_data["max_length"]
        tokens = text_clean.lower().split()[:max_len]
        ids = [vocab.get(t, vocab.get("<UNK>", 1)) for t in tokens]
        if len(ids) < max_len:
            ids += [vocab.get("<PAD>", 0)] * (max_len - len(ids))
        x_tensor = torch.tensor([ids], dtype=torch.long)
        with torch.no_grad():
            logits = model(x_tensor)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        p_neg, p_neu, p_pos = float(probs[0]), float(probs[1]), float(probs[2])
        pred_id = int(np.argmax(probs))
        conf = float(probs[pred_id])
    elif algorithm == "Transformer (PhoBERT)":
        phobert_data = loaded_models.get(algorithm)
        if phobert_data is None:
            raise ValueError("Mô hình PhoBERT chưa được nạp.")
        model = phobert_data["model"]
        tokenizer = phobert_data["tokenizer"]
        inputs = tokenizer(text_clean,return_tensors="pt",truncation=True,max_length=64,padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)[0].cpu().numpy()
        p_neg, p_neu, p_pos = float(probs[0]), float(probs[1]), float(probs[2])
        pred_id = int(np.argmax(probs))
        conf = float(probs[pred_id])
    else:
        raise ValueError(f"Thuật toán không hợp lệ: {algorithm}")
    label_str, badge_class = LABEL_NAMES[pred_id]
    return {
        "algorithm": algorithm,
        "label": label_str,
        "class_name": badge_class,
        "confidence": f"{conf * 100:.1f}%",
        "probs": f"Tích cực: {p_pos * 100:.1f}% | Trung tính: {p_neu * 100:.1f}% | Tiêu cực: {p_neg * 100:.1f}%"
    }
class AppHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB_DIR), **kwargs)
    def do_GET(self):
        if self.path in ("/", ""):
            self.path = "/index.html"
        return super().do_GET()
    def do_POST(self):
        if self.path == "/api/predict":
            try:
                content_len = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_len)
                data = json.loads(body.decode("utf-8"))
                text = data.get("text", "")
                algorithm = data.get("algorithm", "Multinomial Naive Bayes")
                result = predict_sentiment(text, algorithm)
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.end_headers()
                self.wfile.write(json.dumps(result, ensure_ascii=False).encode("utf-8"))
            except Exception as e:
                self.send_response(500)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}, ensure_ascii=False).encode("utf-8"))
            return
        self.send_response(404)
        self.end_headers()
def run_server(port=8000):
    server = HTTPServer(("", port), AppHandler)
    print(f"http://localhost:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()
if __name__ == "__main__":
    port = 8000
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            pass
    run_server(port)
