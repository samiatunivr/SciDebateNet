
import tempfile
from scidebate_demo import Retriever, load_docs

# minimal test: build retriever with tiny docs

def test_retriever_build_and_query():
    docs = [{"id":"d1","text":"alpha beta gamma"}, {"id":"d2","text":"delta epsilon zeta"}]
    r = Retriever()
    r.build(docs)
    res = r.query("alpha")
    assert len(res) >= 1
