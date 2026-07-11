# FactMatch

FactMatch is a fast, async RAG-based fact checker. Paste in any text, and the app extracts factual claims, searches live sources (via Tavily), and judges the evidence to give you a verdict.

### Features
- **Concurrent Processing:** Checks multiple claims simultaneously for fast results.
- **Live Search:** Uses real-time web search instead of relying on stale LLM memory.
- **Clear UI:** Clean and intuitive interface that groups matching and contradicting sources.
- **Free/Mock Mode Available:** Built-in ability to run in a fully mocked mode for cost-free testing.

### Run Locally

```bash
cd backend
pip install -r requirements.txt

# Run the server
uvicorn main:app --reload --port 8000
```
Then open **http://localhost:8000** in your browser.

### Run with Docker

```bash
docker build -t factmatch .
docker run -p 8000:8000 factmatch
```
