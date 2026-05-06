# 🤖 Qwen2.5 Local AI Assistant

A fully local AI chatbot with a clean dark UI — no API keys, no subscriptions, no cloud.

Runs entirely on your machine using open-source Qwen2.5 models.

---

## ✨ Features

- 🧠 Runs **100% locally** (offline after first download)
- ⚡ Automatic setup (installs dependencies for you)
- 💬 Clean **dark-mode chat UI** (Tkinter)
- 🔄 Switch between multiple Qwen2.5 models
- 💾 Models cached locally (download once)
- 🖥️ Works on **CPU, GPU (CUDA), or Apple Silicon**

---

## 📦 Models

| Model | Size | Description |
|------|------|------------|
| Qwen2.5-0.5B | ~400 MB | Fastest ⚡ |
| Qwen2.5-1.5B | ~1.1 GB | Balanced |
| Qwen2.5-3B   | ~2 GB   | Best quality 🧠 |

---

## 🚀 Usage

```bash
python qwen_assistant.py
```

### First Run

- Installs required Python packages automatically
- Downloads the selected model
- May take a few minutes depending on your internet

---

## 🧰 Requirements

- Python **3.9+**
- Internet connection (first run only)

> No manual `pip install` needed — the script handles everything.

---

## 🧠 How It Works

- Uses `transformers` + `torch` to run Qwen models locally
- Automatically detects:
  - CUDA (NVIDIA GPU)
  - MPS (Apple Silicon)
  - CPU fallback
- Stores models in:

```
./model/
```

---

## 🔒 Privacy

- ✅ No API keys
- ✅ No accounts
- ✅ No cloud usage after model download
- ✅ No tracking or telemetry

Everything runs locally on your machine.

---

## ⚠️ Notes

- CPU performance:
  - 0.5B → usable
  - 1.5B → slower
  - 3B → very slow
- First launch may take time due to downloads
- Requires a few GB of disk space

---

## 🖼️ UI

- Dark theme
- Chat-style interface
- Model switcher (top-right)
- `Enter` = send  
- `Shift + Enter` = new line

---

## 📁 Project Structure

```
qwen_assistant.py
/model/
```

---

## 🛠️ Customization

- Add models in `AVAILABLE_MODELS`
- Adjust generation settings (temperature, tokens, etc.)
- Modify UI colors in `PAL`

---

## 📜 License

MIT - free usage.

---

## 🙌 Credits

- https://huggingface.co/Qwen
- https://github.com/huggingface/transformers
- https://pytorch.org
