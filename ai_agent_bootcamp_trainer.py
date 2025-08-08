"""
AI Agent CEO Bootcamp — Trainer Mode App (Streamlit)

Single-file Streamlit application that:
- Presents the 11-day, 110-hour bootcamp as step-by-step lessons
- Lets you configure multiple LLM backends (OpenAI, Azure OpenAI, Ollama, Gemini placeholder)
- Generates hourly learning content by sending a templated prompt to your selected LLM
- Tracks progress (SQLite) and stores generated content
- Exports lesson content to Markdown

Run:
    pip install -r requirements.txt
    streamlit run ai_agent_bootcamp_trainer.py

Requirements (suggested):
    streamlit, openai, requests

Security note: This example stores LLM credentials locally in a .llm_config.json file for convenience. For production use, prefer environment variables or secure secret stores.

This is a starter app — you can extend adapters for other LLMs, add user auth, or host the DB centrally.
"""

import streamlit as st
import sqlite3
import os
import json
import datetime
import textwrap
import requests
from datetime import datetime, timezone
# Optional: import openai if available
try:
    import openai
except Exception:
    openai = None

# -----------------------------
# Bootcamp content (110 topics)
# -----------------------------
TOPICS = [
    # Day 1
    "Agent Fundamentals (types, architecture, autonomy levels)",
    "Key components (LLM, tools, memory, orchestration)",
    "Prompt engineering essentials",
    "Setting up local environment",
    "Building first agent – single tool (API fetcher)",
    "Continuing agent build – integrating with LLM",
    "Adding simple memory",
    "Error handling basics",
    "Deploying locally",
    "Review + checkpoint quiz",
    # Day 2
    "Multi-agent communication models",
    "Role-based agents (planner, executor, reviewer)",
    "Coordination patterns (sequential, parallel, hub-and-spoke)",
    "Handling shared state without conflicts",
    "Implementing planner–executor–reviewer system (Part 1)",
    "Implementing planner–executor–reviewer system (Part 2)",
    "Testing inter-agent trust and verification",
    "Debugging multi-agent workflows",
    "Performance considerations in multi-agent workflows",
    "Review + checkpoint quiz",
    # Day 3
    "Adding APIs & SDKs",
    "Working with databases",
    "Integrating external LLMs and embeddings",
    "Handling file uploads/downloads",
    "SOAP/legacy API handling",
    "Search & RAG pipeline integration",
    "Combining multiple data sources",
    "Testing integrations",
    "Error recovery patterns",
    "Review + checkpoint quiz",
    # Day 4
    "Memory types (short-term, long-term, episodic, semantic)",
    "Vector stores & embeddings",
    "Indexed search for agents",
    "Incremental updates without retraining",
    "Implementing hybrid memory (short + long-term) – Part 1",
    "Implementing hybrid memory – Part 2",
    "Forgetting & pruning strategies",
    "Knowledge aging and versioning",
    "Testing recall accuracy",
    "Review + checkpoint quiz",
    # Day 5
    "Planning algorithms for agents",
    "Self-reflection loops",
    "Error analysis automation",
    "Skill learning via API exploration",
    "Implementing autonomous task re-prioritization – Part 1",
    "Implementing autonomous task re-prioritization – Part 2",
    "Balancing autonomy with human oversight",
    "Detecting goal drift",
    "Sandbox testing for safe learning",
    "Review + checkpoint quiz",
    # Day 6
    "Output filtering & red teaming",
    "Prompt injection defense",
    "Role-based access control for agents",
    "Authentication & signed agent messages",
    "Supply chain security (dependencies)",
    "Monitoring for malicious behavior",
    "Fail-safe shutdown mechanisms",
    "User privacy & compliance (GDPR, HIPAA)",
    "Incident response playbooks",
    "Review + checkpoint quiz",
    # Day 7
    "Evaluation metrics (BLEU, ROUGE, task-specific)",
    "Success/failure tracking",
    "Latency and throughput testing",
    "Regression testing pipelines",
    "Automated evaluation agents – Part 1",
    "Automated evaluation agents – Part 2",
    "A/B testing agents",
    "Data-driven optimization",
    "Performance dashboards",
    "Review + checkpoint quiz",
    # Day 8
    "Horizontal scaling strategies",
    "Multi-region deployments",
    "Cost forecasting models",
    "Load balancing for LLM calls",
    "Distributed memory architectures – Part 1",
    "Distributed memory architectures – Part 2",
    "Error rate thresholds & auto-recovery",
    "Monitoring and alerting setup",
    "Disaster recovery drills",
    "Review + checkpoint quiz",
    # Day 9
    "Market analysis",
    "Problem–solution fit validation",
    "Pricing strategies",
    "Subscription vs. transaction models",
    "Cost vs. revenue modeling – Part 1",
    "Cost vs. revenue modeling – Part 2",
    "Marketing channels for agent products",
    "Case studies of successful agent businesses",
    "Preparing investor pitch decks",
    "Review + checkpoint quiz",
    # Day 10
    "Bias detection techniques",
    "Explainable AI (XAI) for agents",
    "Cultural & linguistic sensitivity",
    "Accessibility compliance",
    "Ethics playbook creation – Part 1",
    "Ethics playbook creation – Part 2",
    "Community feedback loops",
    "Transparency reporting",
    "Balancing profit and responsibility",
    "Review + checkpoint quiz",
    # Day 11
    "Final project planning & scope definition",
    "Building final project – Phase 1",
    "Building final project – Phase 2",
    "Building final project – Phase 3",
    "Code review & evaluation – Part 1",
    "Code review & evaluation – Part 2",
    "Load testing & security validation – Part 1",
    "Load testing & security validation – Part 2",
    "Business pitch preparation & delivery",
    "Graduation & certification",
]

PROMPT_TEMPLATE = textwrap.dedent(
    """
You are an expert AI Agent Trainer conducting Hour {hour} of Day {day} in an 11-day AI Agent CEO Bootcamp. 
The topic for this hour is: "{topic}".

Your task:
1. Explain the concept in simple, clear language with real-world analogies.
2. Provide at least one diagram description or visual analogy idea.
3. Give a step-by-step guided demo or practical exercise relevant to the topic.
4. Include a starter code template in Python (if applicable) or pseudocode.
5. Give 3 mini practice challenges of increasing difficulty.
6. Include a 3-question quiz to check understanding.
7. Provide 1 mini-project or applied scenario students can attempt after this hour.
8. End with a “Pro Tip” for real-world application.

Style: conversational but precise, use bullet points, and ensure every section is actionable.
"""
)

# -----------------------------
# Simple local config helpers
# -----------------------------
CONFIG_PATH = ".llm_config.json"
DB_PATH = "bootcamp_progress.db"
EXPORT_DIR = "exports"

os.makedirs(EXPORT_DIR, exist_ok=True)


def load_config():
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_config(cfg):
    with open(CONFIG_PATH, "w") as f:
        json.dump(cfg, f)


# -----------------------------
# DB helpers
# -----------------------------

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS steps (
            id INTEGER PRIMARY KEY,
            day INTEGER,
            hour INTEGER,
            title TEXT,
            topic TEXT,
            prompt TEXT,
            status TEXT DEFAULT 'not_started',
            content TEXT,
            last_updated TEXT
        )
        """
    )

    # Insert steps if not present
    c.execute("SELECT COUNT(*) FROM steps")
    count = c.fetchone()[0]
    if count == 0:
        idx = 0
        for day in range(1, 12):
            for hour in range(1, 11):
                topic = TOPICS[idx]
                title = f"Day {day} - Hour {hour}: {topic}"
                prompt = PROMPT_TEMPLATE.format(day=day, hour=hour, topic=topic)
                c.execute(
                    "INSERT INTO steps (day, hour, title, topic, prompt) VALUES (?, ?, ?, ?, ?)",
                    (day, hour, title, topic, prompt),
                )
                idx += 1
        conn.commit()
    conn.close()


def get_step(day, hour):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, day, hour, title, topic, prompt, status, content, last_updated FROM steps WHERE day=? AND hour=?", (day, hour))
    row = c.fetchone()
    conn.close()
    if row:
        keys = ["id", "day", "hour", "title", "topic", "prompt", "status", "content", "last_updated"]
        return dict(zip(keys, row))
    return None


def update_step_content(step_id, content, status="completed"):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "UPDATE steps SET content=?, status=?, last_updated=? WHERE id=?",
        (content, status, datetime.now(timezone.utc).isoformat(), step_id),
    )
    conn.commit()
    conn.close()


def list_steps(status_filter=None):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    if status_filter:
        c.execute("SELECT id, day, hour, title, status, last_updated FROM steps WHERE status=? ORDER BY day, hour", (status_filter,))
    else:
        c.execute("SELECT id, day, hour, title, status, last_updated FROM steps ORDER BY day, hour")
    rows = c.fetchall()
    conn.close()
    return rows


# -----------------------------
# LLM Adapters
# -----------------------------
class BaseAdapter:
    def __init__(self, config):
        self.config = config

    def generate(self, prompt):
        raise NotImplementedError("Adapter must implement generate()")


class OpenAIAdapter(BaseAdapter):
    def __init__(self, config):
        super().__init__(config)
        if openai is None:
            raise RuntimeError("Please `pip install openai` to use the OpenAI adapter")
        # config: {api_key, model, api_base(optional), api_type(optional), api_version(optional)}
        key = config.get("api_key")
        openai.api_key = key
        if config.get("api_type") == "azure":
            # Azure config expects api_base, api_version and deployment name as model
            openai.api_type = "azure"
            openai.api_base = config.get("api_base")
            openai.api_version = config.get("api_version") or "2023-05-15"
            # For azure, the model param should be the deployment name; we'll pass as model in the call

    def generate(self, prompt):
        cfg = self.config
        model = cfg.get("model") or "gpt-4o"
        try:
            if cfg.get("api_type") == "azure":
                # Azure: use deployment name as model
                response = openai.ChatCompletion.create(
                    deployment_id=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                    max_tokens=1200,
                )
            else:
                # Standard OpenAI API call
                response = openai.ChatCompletion.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                    max_tokens=1200,
                )

            # New response format
            return response.choices[0].message.content  # Extract the generated content

        except Exception as e:
            return f"<Error from OpenAI adapter: {e}>"
class OllamaAdapter(BaseAdapter):
    def __init__(self, config):
        super().__init__(config)
        # Ollama usually runs locally: http://localhost:11434
        self.url = config.get("url") or "http://localhost:11434/api/generate"
        self.model = config.get("model") or "llama2"

    # def generate(self, prompt):
    #     try:
    #         payload = {"model": self.model, "prompt": prompt}
    #         r = requests.post(self.url, json=payload, timeout=60)
    #         r.raise_for_status()
    #         data = r.json()
    #         # Adjust parsing depending on Ollama response format
    #         return data.get("text") or json.dumps(data)
    #     except Exception as e:
    #         return f"<Error from Ollama adapter: {e}>"
    def generate(self, prompt):
        try:
            payload = {"model": self.model, "prompt": prompt, "stream": False}
            r = requests.post(self.url, json=payload, timeout=120)
            r.raise_for_status()
            data = r.json()
            return data.get("response") or data.get("text") or json.dumps(data)
        except Exception as e:
            return f"<Error from Ollama adapter: {e}>"



class GeminiAdapter(BaseAdapter):
    def __init__(self, config):
        super().__init__(config)
        # Placeholder - Gemini often requires Google Cloud setup and special client libs
        self.url = config.get("url")

    def generate(self, prompt):
        # Provide a helpful message instructing the user how to implement
        return "<Gemini adapter not implemented. Please add your Gemini HTTP call or use OpenAI/Azure/Ollama.>"


# Factory
ADAPTERS = {
    "openai": OpenAIAdapter,
    "ollama": OllamaAdapter,
    "gemini": GeminiAdapter,
}


def get_adapter(provider_key, cfg):
    cls = ADAPTERS.get(provider_key)
    if not cls:
        raise RuntimeError(f"Unknown provider: {provider_key}")
    return cls(cfg)


# -----------------------------
# Streamlit UI
# -----------------------------

st.set_page_config(page_title="AI Agent CEO Bootcamp Trainer", layout="wide")
init_db()

cfg = load_config()

with st.sidebar:
    st.title("Bootcamp — Settings")
    provider = st.selectbox("LLM Provider", options=["openai", "azure_openai", "ollama", "gemini"], index=0)

    if provider == "openai":
        st.subheader("OpenAI / Chat Models")
        cfg['openai_api_key'] = st.text_input("OpenAI API key", value=cfg.get('openai_api_key', ''), type="password")
        cfg['openai_model'] = st.text_input("Model (e.g. gpt-4o-mini)", value=cfg.get('openai_model', 'gpt-4o'))
        if st.button("Save OpenAI config"):
            save_config(cfg)
            st.success("Saved")
    elif provider == "azure_openai":
        st.subheader("Azure OpenAI")
        cfg['azure_api_key'] = st.text_input("Azure API key", value=cfg.get('azure_api_key', ''), type="password")
        cfg['azure_api_base'] = st.text_input("Azure API base (e.g. https://your-resource.openai.azure.com)", value=cfg.get('azure_api_base', ''))
        cfg['azure_deployment'] = st.text_input("Deployment name (model deployment)", value=cfg.get('azure_deployment', ''))
        cfg['azure_api_version'] = st.text_input("API version", value=cfg.get('azure_api_version', '2023-05-15'))
        if st.button("Save Azure config"):
            save_config(cfg)
            st.success("Saved")
    elif provider == "ollama":
        st.subheader("Ollama (local)")
        cfg['ollama_url'] = st.text_input("Ollama URL", value=cfg.get('ollama_url', 'http://localhost:11434/api/generate'))
        cfg['ollama_model'] = st.text_input("Ollama model name", value=cfg.get('ollama_model', 'llama2'))
        if st.button("Save Ollama config"):
            save_config(cfg)
            st.success("Saved")
    elif provider == "gemini":
        st.subheader("Gemini (placeholder)")
        st.markdown("Gemini adapter is a placeholder. Please add your API details in .llm_config.json or extend the GeminiAdapter in the code.")

    st.markdown("---")
    st.markdown("**Progress**")
    rows = list_steps()
    completed = len([r for r in rows if r[4] == 'completed'])
    total = len(rows)
    st.progress(int(completed / total * 100) if total else 0)
    st.write(f"{completed} / {total} completed")


# Main UI — top controls
col1, col2 = st.columns([1, 3])

with col1:
    st.header("AI Agent CEO Bootcamp")
    day = st.number_input("Day", min_value=1, max_value=11, value=1)
    hour = st.number_input("Hour", min_value=1, max_value=10, value=1)

    if st.button("Load step"):
        st.session_state['trigger_rerun'] = True

    st.markdown("---")
    st.button("Export all completed as Markdown", on_click=None)
    # Footer: Created by...
    st.markdown(
        """
        ---
        **Created by [Abhinav Kumar](https://github.com/akumar261089)**  
        """
    )

with col2:
    step = get_step(day, hour)
    if not step:
        st.error("Step not found — try different day/hour")
    else:
        st.subheader(step['title'])
        st.caption(f"Status: {step['status']} — Last updated: {step['last_updated']}")

        st.markdown("### Generated Prompt")
        st.code(step['prompt'], language='')

        # Action buttons
        st.markdown("---")
        gen_col1, gen_col2, gen_col3, gen_col4 = st.columns([1,1,1,1])
        with gen_col1:
            generate = st.button("Generate Content from LLM")
        with gen_col2:
            mark_complete = st.button("Mark Complete")
        with gen_col3:
            export_md = st.button("Export this step as Markdown")
        with gen_col4:
            import_md = st.button("Load from Markdown")

        # Build adapter config dict based on provider selection
        adapter_cfg = {}
        try:
            if provider == 'openai':
                adapter_cfg = {
                    'api_key': cfg.get('openai_api_key'),
                    'model': cfg.get('openai_model')
                }
                adapter = OpenAIAdapter({'api_key': cfg.get('openai_api_key'), 'model': cfg.get('openai_model')})
            elif provider == 'azure_openai':
                adapter = OpenAIAdapter({
                    'api_key': cfg.get('azure_api_key'),
                    'api_type': 'azure',
                    'api_base': cfg.get('azure_api_base'),
                    'api_version': cfg.get('azure_api_version'),
                    'model': cfg.get('azure_deployment'),
                })
            elif provider == 'ollama':
                adapter = OllamaAdapter({'url': cfg.get('ollama_url'), 'model': cfg.get('ollama_model')})
            else:
                adapter = GeminiAdapter({})
        except Exception as e:
            st.error(f"Error creating adapter: {e}")
            adapter = None

        if generate:
            if adapter is None:
                st.error("No valid adapter configured.")
            else:
                with st.spinner("Generating content. This may take a moment..."):
                    result = adapter.generate(step['prompt'])
                    update_step_content(step['id'], result, status='completed')
                    st.success("Generated and saved")
                    st.session_state['trigger_rerun'] = True

        if mark_complete:
            update_step_content(step['id'], step.get('content') or '', status='completed')
            st.success("Marked complete")
            st.session_state['trigger_rerun'] = True

        if export_md:
            filename = os.path.join(EXPORT_DIR, f"Day{day}_Hour{hour}.md")
            content = step.get('content') or ""
            md = f"# {step['title']}\n\n## Topic:\n{step['topic']}\n\n## Prompt:\n{step['prompt']}\n\n## Generated Content:\n\n{content}\n"
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(md)
            st.success(f"Exported to {filename}")
        if import_md:
            filename = os.path.join(EXPORT_DIR, f"Day{day}_Hour{hour}.md")

            if os.path.exists(filename):
                try:
                    with open(filename, "r", encoding="utf-8") as f:
                        md_text = f.read()

                    # Crude parse: everything after '## Generated Content:' is lesson content
                    if "## Generated Content:" in md_text:
                        imported_content = md_text.split("## Generated Content:")[1].strip()
                        update_step_content(step['id'], imported_content, status='completed')
                        st.success("Imported from Markdown and saved to DB.")
                        st.session_state['trigger_rerun'] = True
                    else:
                        st.error("Invalid markdown format for import.")
                except Exception as e:
                    st.error(f"Error reading markdown file: {e}")
            else:
                st.error(f"Markdown file not found: {filename}")


        st.markdown("---")
        st.markdown("### Generated Content")
        if step.get('content'):
            st.markdown(step['content'])
        else:
            st.info("No generated content yet. Click 'Generate Content from LLM' to create it.")


# Navigation and batch features
st.sidebar.markdown("---")
st.sidebar.header("Navigation & Batch")
if st.sidebar.button("Go to next hour"):
    next_hour = hour + 1
    next_day = day
    if next_hour > 10:
        next_hour = 1
        next_day = min(11, day + 1)
    st.session_state['nav_day'] = next_day
    st.session_state['nav_hour'] = next_hour
    st.session_state['trigger_rerun'] = True

if st.sidebar.button("Go to previous hour"):
    prev_hour = hour - 1
    prev_day = day
    if prev_hour < 1:
        prev_hour = 10
        prev_day = max(1, day - 1)
    st.session_state['nav_day'] = prev_day
    st.session_state['nav_hour'] = prev_hour
    st.session_state['trigger_rerun'] = True

st.sidebar.markdown("---")
st.sidebar.subheader("Batch generate (selected range)")
batch_days = st.sidebar.multiselect("Days to include", options=list(range(1, 12)), default=[1])
batch_button = st.sidebar.button("Generate batch for selected days")

if batch_button:
    # Fetch steps in selected days
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    placeholders = ','.join('?' * len(batch_days))
    c.execute(f"SELECT id, day, hour, title, prompt FROM steps WHERE day IN ({placeholders}) ORDER BY day, hour", tuple(batch_days))
    rows = c.fetchall()
    conn.close()

    st.sidebar.write(f"Found {len(rows)} steps. Generating...")
    adapter_ok = True
    try:
        # reuse adapter from above
        pass
    except Exception as e:
        st.sidebar.error(f"Adapter error: {e}")
        adapter_ok = False

    if adapter_ok:
        # run sequentially (explicit — no background threads)
        for rid, rday, rhour, rtitle, rprompt in rows:
            st.sidebar.write(f"Generating Day {rday} Hour {rhour} — {rtitle}")
            try:
                text = adapter.generate(rprompt)
                update_step_content(rid, text, status='completed')
            except Exception as e:
                st.sidebar.error(f"Failed for Day {rday} Hour {rhour}: {e}")
        st.sidebar.success("Batch generation finished. Refresh main view to see updates.")


# Progress table
st.markdown("## Progress Tracker")
rows = list_steps()
if rows:
    import pandas as pd

    df = pd.DataFrame(rows, columns=["id", "day", "hour", "title", "status", "last_updated"])[:]
    st.dataframe(df)
    st.download_button("Export progress CSV", df.to_csv(index=False), file_name="bootcamp_progress.csv")
else:
    st.info("No steps in DB — you probably need to initialize the DB.")


# Help / instructions
st.markdown("---")
st.header("Quick Start & Tips")
st.markdown(
    """
- Configure your chosen LLM in the sidebar (OpenAI is implemented; Azure uses the same adapter with `api_type='azure'` and the deployment name as the model).
- For Ollama, make sure the Ollama local runtime is running and the URL matches.
- Click "Generate Content from LLM" to create the hour's lesson. The app stores generated material and marks the step completed.
- Use batch generation carefully — it will sequentially call your LLM for every step you select and may incur costs.

Security note: API keys stored in `.llm_config.json` are not encrypted. For production, use environment variables or a secret manager.
"""
)

# Footer
st.sidebar.markdown("---")
st.sidebar.write("Built-in trainer-mode prototype — customize it for your org.")


# End of file
