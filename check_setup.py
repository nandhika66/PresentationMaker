# Run this first if anything fails. It shows which Python you are using,
# whether every library loads, and whether MiniLM actually runs.
import importlib
import sys

print(f"Python executable: {sys.executable}")
print(f"Python version:    {sys.version.split()[0]}")

in_venv = sys.prefix != sys.base_prefix
print(f"Inside a venv:     {'YES' if in_venv else 'NO  <-- activate .venv first!'}")

packages = {
    "pptx": "python-pptx",
    "numpy": "numpy",
    "rank_bm25": "rank-bm25",
    "onnxruntime": "onnxruntime",
    "tokenizers": "tokenizers",
    "huggingface_hub": "huggingface-hub",
    "google.genai": "google-genai",
    "dotenv": "python-dotenv",
}

ok = True
print()
for module, name in packages.items():
    try:
        mod = importlib.import_module(module)
        print(f"  OK   {name:<18} {getattr(mod, '__version__', '')}")
    except Exception as error:
        ok = False
        print(f"  FAIL {name:<18} {type(error).__name__}: {error}")

print("\nTesting MiniLM (first time downloads ~90 MB)...")
try:
    from embedder import MiniLMEmbedder

    vector = MiniLMEmbedder().encode("consistency in distributed databases")
    print(f"  OK   MiniLM works, vector size {vector.shape[0]}")
except Exception as error:
    ok = False
    print(f"  FAIL MiniLM: {type(error).__name__}: {error}")

print("\nAll good!" if ok else "\nSomething failed - see README.md > Troubleshooting")
