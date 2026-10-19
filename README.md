# PresentationMaker – Phase 1 (no OCR)

Reads your PPTX files → builds a searchable knowledge base (BM25 + MiniLM hybrid
search) → sends the most relevant slides to Gemini → saves a new PPTX.

**No PyTorch.** MiniLM (all-MiniLM-L6-v2) runs through its official ONNX export on
Microsoft's `onnxruntime`. It produces the same vectors, has a smaller install, and isn't
blocked by Windows Smart App Control.

## Project files

| File | What it does |
|---|---|
| `config.py` | All paths and model names in one place |
| `read_ppt.py` | **Step 1** – extracts text from every PPTX in `input_ppts/` → `output/slide_records.json` |
| `embedder.py` | MiniLM via onnxruntime (replaces sentence-transformers / PyTorch) |
| `build_embeddings.py` | **Step 2** – MiniLM embeddings → `output/slide_embeddings.npy` |
| `hybrid_search.py` | Keyword + BM25 + MiniLM combined with RRF (run it alone for interactive search) |
| `generate_content.py` | Sends retrieved slides + your request to Gemini, gets structured JSON |
| `generate_presentation.py` | **Step 3** – the main program, creates the new PPTX |
| `evaluate_search.py` | Optional – Precision@5 / Recall@5 / MRR using `data/evaluation_queries.json` |
| `test_gemini.py` | Optional – checks your Gemini API key |
| `check_setup.py` | Checks your Python, venv and libraries (run this if anything fails) |

## Installation (Windows, one time)

Use **Python 3.11** (3.10–3.12 also work). Open **Command Prompt** in this folder:

```bat
:: 1. Create a fresh virtual environment INSIDE this folder
py -3.11 -m venv .venv

:: 2. Activate it  (you should see (.venv) at the start of the line)
.venv\Scripts\activate

:: 3. Install packages
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

:: 4. Verify everything loads
python check_setup.py
```

Or just double-click **`setup.bat`**, which runs all of the above.

PowerShell instead of Command Prompt: activate with `.venv\Scripts\Activate.ps1`.
If it says scripts are disabled, run once:
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

### API key
Open `.env` and set your key (see `.env.example`):
```
GEMINI_API_KEY=your-key-here
```

## Running

Always activate the venv first (`.venv\Scripts\activate`), then:

```bat
python read_ppt.py               :: Step 1 – run again whenever you add/change PPTs
python build_embeddings.py       :: Step 2 – run again after Step 1
python generate_presentation.py  :: Step 3 – ask for a presentation
```

Optional:
```bat
python hybrid_search.py      :: interactive search only
python evaluate_search.py    :: retrieval metrics
python test_gemini.py        :: test the API key
```

The first run of Step 2 (or `check_setup.py`) downloads the MiniLM ONNX model
(~90 MB) from Hugging Face. After that it is cached.

## Troubleshooting

**`WinError 4551: An Application Control policy has blocked this file`**
Windows Smart App Control (or a company/college device policy) blocked a DLL.
That's why the project no longer uses PyTorch. If it ever blocks onnxruntime too,
the program does not crash: it prints a warning and searches with keyword + BM25
only. Smart App Control is under Windows Security → App & browser control. On a
managed laptop, ask IT.

**"Disabling PyTorch…" / old venv issues**
You are running a different Python than the venv's. This is what broke the old
project: the venv was created in the `ppt-generation` folder and then the
project moved to `ppt-generation-backup`. A Windows venv stores its original
folder path, so after the move `activate` silently used the system Python and
its old PyTorch.
Fix: delete `.venv`, then repeat the installation steps.
**Do not move or rename the project folder after creating `.venv`.**
`python check_setup.py` shows which Python is being used.

**`DLL load failed` for onnxruntime (WinError 126 / 1114)**
Install the Microsoft Visual C++ Redistributable (x64), restart the terminal,
and try again.

**"slides but … embeddings"**
You re-ran `read_ppt.py` without re-running `build_embeddings.py`.

**Gemini 503 / "model overloaded"**
That's a temporary server error, so wait a minute and retry. You can change the model in
`config.py` (`GEMINI_MODEL`).

## Adding OCR later
Keep OCR as a separate step that writes an `ocr_text` field into
`slide_records.json` **between Step 1 and Step 2**, then rebuild embeddings.
Never import it from the search or generation scripts.
