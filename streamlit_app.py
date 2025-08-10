"""
Streamlit UI for browsing debates and triggering new debates interactively.
Run with: streamlit run streamlit_app.py
"""
import streamlit as st
import json
from pathlib import Path
import os

from scidebate_demo import Retriever, LLM, run_debate, load_docs

st.set_page_config(page_title="SciDebateNet", layout="wide")
st.title("SciDebateNet — Debate-driven research assistant")

DATA_DIR = Path("./papers")

if st.sidebar.button('Run new debate'):
    with st.spinner('Running debate...'):
        docs = load_docs(DATA_DIR)
        retr = Retriever()
        retr.build(docs)
        llm = LLM()
        seed = st.sidebar.text_area('Seed prompt', 'Explore drug repurposing hypotheses for cytokine-mediated disease.')
        result = run_debate(seed, retr, llm, rounds=2)
        # save and show
        out_path = Path(f"ui_debate_{int(time.time())}.json")
        out_path.write_text(json.dumps(result, indent=2))
        st.success('Debate finished. Scroll to view results.')
        st.json(result)

st.sidebar.markdown('### Existing debate results')
for f in sorted(Path('.').glob('debate_result_*.json'))[-5:]:
    if st.sidebar.button(f.name):
        st.json(json.loads(f.read_text()))

```
