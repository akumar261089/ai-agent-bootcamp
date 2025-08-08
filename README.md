# 🚀 AI Agent Bootcamp Trainer

*A complete, interactive Trainer Mode app for mastering AI Agent development — from fundamentals to advanced security.*

![Streamlit](https://img.shields.io/badge/Framework-Streamlit-FF4B4B?style=for-the-badge)
![OpenAI](https://img.shields.io/badge/LLM-OpenAI-blue?style=for-the-badge)
![Azure](https://img.shields.io/badge/LLM-Azure%20OpenAI-0078D4?style=for-the-badge)
![Ollama](https://img.shields.io/badge/LLM-Ollama-black?style=for-the-badge)
![Gemini](https://img.shields.io/badge/LLM-Gemini-4285F4?style=for-the-badge)

---

## 📖 Overview

**AI Agent Bootcamp Trainer** is a step-by-step, 11-day, hour-by-hour learning platform to master AI Agent systems.
You can:

* Select **your preferred LLM** (OpenAI, Azure OpenAI, Ollama, Gemini\*)
* Follow structured **hour-by-hour prompts** for theory, practicals, and projects
* **Track progress** for each step
* Export notes as Markdown
* Batch-generate learning materials

> *Gemini adapter placeholder included — can be extended.*

---

## ✨ Features

* 📅 **110 structured steps** (Day 1 → Day 11) with pre-built prompts
* ⚙️ **Multi-LLM support**: OpenAI, Azure OpenAI, Ollama, Gemini
* 📊 **Progress tracking** with SQLite
* 📝 **One-click LLM content generation**
* 📦 **Markdown export** per step
* 🔄 **Batch-generate** for multiple steps at once
* 🎨 Clean **Streamlit UI** with a sidebar config

---

## 🖥 Demo Screenshot

![UI Screenshot](docs/ui-screenshot.png)
*(Replace with actual screenshot after running)*

---

## 📂 Project Structure

```
ai_agent_bootcamp_trainer.py    # Main Streamlit app
llm_config.json                 # LLM settings (created at runtime)
bootcamp.db                     # SQLite DB for progress (created at runtime)
README.md                       # This file
requirements.txt                # Python dependencies
```

---

## 🚀 Installation

### 1️⃣ Clone the repo

```bash
git clone https://github.com/yourusername/ai-agent-bootcamp-trainer.git
cd ai-agent-bootcamp-trainer
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the app

```bash
streamlit run ai_agent_bootcamp_trainer.py
```

---

## ⚙️ LLM Configuration

From the sidebar:

* **Provider**: OpenAI / Azure OpenAI / Ollama / Gemini
* **Model Name**: Example: `gpt-4o-mini`, `llama2`, `gemini-pro`
* **API Key**: Your provider’s key
* **Endpoint** (if applicable): For Azure OpenAI & Ollama

LLM settings are stored locally in `.llm_config.json`.

---

## 🗂 Progress Tracking

* All progress is stored in `bootcamp.db`
* Each step has: status, notes, generated content, timestamps
* **Mark Complete** to save your work
* Export **CSV** or **Markdown** for review

---

## 🛠 Development

To modify:

* Update **bootcamp schedule & prompts** inside `ai_agent_bootcamp_trainer.py`
* Extend `llm_adapters` for new models or custom APIs

---

## 📦 Deployment

To deploy:

```bash
pip install -r requirements.txt
streamlit run ai_agent_bootcamp_trainer.py
```

Or package with Docker:

```bash
docker build -t ai-agent-bootcamp .
docker run -p 8501:8501 ai-agent-bootcamp
```

---

## 📋 Roadmap

* [ ] Add **Gemini API adapter**
* [ ] Add **user accounts** with authentication
* [ ] Sync progress to cloud DB
* [ ] Advanced analytics dashboard

---

## 📜 License

MIT License. See [LICENSE](LICENSE) for details.

---
