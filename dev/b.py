import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from matplotlib.animation import FuncAnimation
import os

plt.rcParams["font.family"] = "Meiryo"

SAVE_PATH = "transcriptions.txt"

tokenizer = AutoTokenizer.from_pretrained(
    "Mizuiro-sakura/luke-japanese-large-sentiment-analysis-wrime"
)
model = AutoModelForSequenceClassification.from_pretrained(
    "Mizuiro-sakura/luke-japanese-large-sentiment-analysis-wrime"
)

emotions = ["喜び", "悲しみ", "期待", "驚き", "怒り", "恐れ", "嫌悪", "信頼", "喜び"]


def get_emotion_probs(text):
    token = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=512, padding="max_length"
    )
    output = model(**token)
    normalized_logits = (output.logits - torch.min(output.logits)) / (
        torch.max(output.logits) - torch.min(output.logits)
    )
    probs = normalized_logits.squeeze().tolist()
    probs.append(probs[0])
    return probs


fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
ax.set_ylim(0, 1)
theta = np.linspace(0, 2 * np.pi, len(emotions), endpoint=True)
(line,) = ax.plot(theta, [0] * len(emotions))
ax.set_xticks(theta)
ax.set_xticklabels(emotions)

last_read_line = 0
last_mtime = os.path.getmtime(SAVE_PATH)  # 最後に確認したファイルの修正時間


def update(frame):
    global last_read_line, last_mtime

    current_mtime = os.path.getmtime(SAVE_PATH)

    if current_mtime != last_mtime:  # ファイルが更新された場合のみ
        with open(SAVE_PATH, "r", encoding="utf-8") as file:
            lines = file.readlines()
            if last_read_line < len(lines):
                text = lines[-1].strip()
                emotion_probs = get_emotion_probs(text)
                line.set_ydata(emotion_probs)
                last_read_line = len(lines)
                last_mtime = current_mtime  # タイムスタンプを更新

                # 感情と確率を表示
                for emotion, prob in zip(emotions, emotion_probs):
                    print(f"{emotion}: {prob:.4f}")
    return (line,)


MAX_FRAMES = 100

ani = FuncAnimation(fig, update, repeat=True, blit=True, save_count=MAX_FRAMES)

plt.show()
