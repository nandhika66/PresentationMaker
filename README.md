# PresentationMaker

**An AI-powered PowerPoint presentation generation system using hybrid retrieval, semantic search, and large language models.**

## 1. Overview

PresentationMaker is an AI-powered application that automatically generates PowerPoint presentations from user-provided PowerPoint files.

Instead of generating presentations entirely from general-purpose AI knowledge, the system retrieves relevant information from uploaded presentations and uses that information as the basis for generating new slides.

Users can upload multiple PPTX files, describe the presentation they want, specify the number of slides, target audience, and presentation objective, and generate a new PowerPoint presentation.

The application combines traditional information retrieval, semantic search, image retrieval, evidence selection, and LLM-based content generation to produce relevant and structured presentations.

## 2. Key Features

- **Multiple PowerPoint Uploads:** Upload multiple PPTX files as source material.
- **Slide Content Extraction:** Extract slide titles, text, and other supported presentation content.
- **Semantic Search:** Retrieve relevant slides using MiniLM embeddings.
- **BM25 Retrieval:** Identify relevant slides through keyword-based ranking.
- **Hybrid Search:** Combine BM25 and semantic retrieval using Reciprocal Rank Fusion (RRF).
- **Image Extraction and Retrieval:** Extract images from source presentations and retrieve relevant images using image metadata and embeddings.
- **Evidence Selection:** Apply Maximal Marginal Relevance (MMR) to select relevant and diverse source slides.
- **AI-Powered Content Generation:** Use Google's Gemini API to generate structured slide content.
- **PowerPoint Generation:** Create a new PPTX presentation containing generated content and relevant original images.
- **Streamlit Interface:** Provide a user-friendly interface for uploading files and configuring presentation requirements.
- **Isolated Processing:** Use a separate temporary workspace for each generation request to prevent uploaded files and generated data from being mixed across requests.

## 3. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core application and pipeline development |
| Streamlit | Web application interface |
| python-pptx | PowerPoint extraction and generation |
| NumPy | Numerical operations and embedding storage |
| MiniLM (all-MiniLM-L6-v2) | Semantic text embeddings |
| ONNX Runtime | Local embedding model inference |
| Tokenizers | Text tokenization |
| Hugging Face Hub | Downloading the embedding model |
| rank-bm25 | BM25-based keyword retrieval |
| Google Gemini API | AI-based slide content generation |
| python-dotenv | Environment variable and API key management |
| Pillow (PIL) | Image processing |

## 4. System Architecture

The application follows a multi-stage presentation generation pipeline.

```text
                 USER
                   |
                   v
          STREAMLIT INTERFACE
                   |
          Upload PPTX Files
          Enter Requirements
                   |
                   v
          TEMPORARY WORKSPACE
                   |
                   v
       POWERPOINT CONTENT EXTRACTION
               read_ppt.py
                   |
                   v
          TEXT EMBEDDING GENERATION
             build_embeddings.py
                   |
                   v
           IMAGE RECORD EXTRACTION
           build_image_records.py
                   |
                   v
          IMAGE EMBEDDING GENERATION
          build_image_embeddings.py
                   |
                   v
             HYBRID RETRIEVAL
            BM25 + MiniLM + RRF
                   |
                   v
           EVIDENCE SELECTION
                  MMR
                   |
                   v
        TEXT AND IMAGE CLASSIFICATION
                   |
             +-----+------+
             |            |
             v            v
        TEXT SOURCES   IMAGE SOURCES
             |            |
             v            v
        GEMINI API    IMAGE RETRIEVAL
             |            |
             v            |
       GENERATED SLIDE    |
          CONTENT         |
             |            |
             +-----+------+
                   |
                   v
          POWERPOINT GENERATION
                   |
                   v
             OUTPUT PPTX
                   |
                   v
           DOWNLOAD PRESENTATION
                   |
                   v
          TEMPORARY FILE CLEANUP
```

## 5. Project Structure

```text
PresentationMaker/
│
├── app.py
├── config.py
│
├── read_ppt.py
├── build_embeddings.py
├── build_image_records.py
├── build_image_embeddings.py
│
├── embedder.py
├── hybrid_search.py
├── search_images.py
│
├── evidence_selection.py
├── generate_content.py
├── generate_presentation.py
│
├── input_ppts/
├── output/
├── data/
│
├── .env
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

### Main Python Files

| File | Responsibility |
|---|---|
| `app.py` | Streamlit application, file uploads, user inputs, pipeline execution, and presentation download |
| `config.py` | Central configuration for project paths, workspaces, embedding settings, and Gemini model |
| `read_ppt.py` | Extract content and metadata from uploaded PowerPoint presentations |
| `build_embeddings.py` | Generate MiniLM embeddings for extracted slide records |
| `build_image_records.py` | Extract image records and associate images with their source slides |
| `build_image_embeddings.py` | Generate embeddings for extracted image records |
| `embedder.py` | Load and run the MiniLM model using ONNX Runtime |
| `hybrid_search.py` | Perform BM25 and semantic retrieval and combine rankings using RRF |
| `search_images.py` | Retrieve relevant images using image metadata, keyword search, BM25, and semantic similarity |
| `evidence_selection.py` | Select relevant and diverse source slides using MMR |
| `generate_content.py` | Prepare source-grounded prompts and generate structured slide content using Gemini |
| `generate_presentation.py` | Coordinate retrieval, evidence selection, image insertion, and PPTX generation |

## 6. How It Works

### Step 1: PowerPoint Content Extraction

The uploaded PowerPoint files are processed using `python-pptx`.

The extraction process collects supported slide information, including:

- Presentation title and slide titles
- Text elements
- Slide metadata
- Speaker notes and tables where supported by the extraction logic
- Original source presentation and slide references

The extracted information is saved as structured JSON records.

### Step 2: Text Embedding Generation

The extracted slide records are converted into numerical vectors using:

**Model:** `sentence-transformers/all-MiniLM-L6-v2`

The model produces 384-dimensional embeddings.

The implementation uses ONNX Runtime for local CPU inference, with tokenization, mean pooling, and L2 normalization.

The generated embeddings are saved as NumPy arrays.

### Step 3: Image Extraction and Indexing

Images are extracted from the uploaded PowerPoint files.

Image records retain information connecting each extracted image to its source presentation and slide.

Image embeddings are generated to support semantic image retrieval.

### Step 4: Hybrid Retrieval

The system retrieves relevant source slides using:

**BM25:** Identifies keyword-level relevance.

**MiniLM:** Captures semantic similarity between the user query and slide content.

**Reciprocal Rank Fusion (RRF):** Combines the rankings produced by different retrieval methods.

This helps retrieve content even when the query uses different wording from the original slides.

### Step 5: Evidence Selection

The retrieved slides are passed through Maximal Marginal Relevance (MMR).

MMR balances:

- Relevance to the user's query
- Diversity among selected slides
- Reduction of repetitive information

This helps select a useful set of source slides for presentation generation.

### Step 6: Content Generation

The selected evidence is classified into text-bearing slides, image-only slides, and empty slides.

Text-bearing source material is provided to Google's Gemini API.

Gemini generates structured slide content in JSON format, following the requested:

- Presentation topic
- Number of slides
- Target audience
- Presentation objective

The generation prompt requires factual statements to be supported by source identifiers from the retrieved material.

### Step 7: Image Retrieval and PowerPoint Generation

Relevant original images are retrieved using the image search pipeline.

The PowerPoint generation module combines generated slide content with relevant source images and creates a new `.pptx` file.

### Step 8: Download and Cleanup

The generated presentation is read into memory and made available through Streamlit's download button.

The temporary workspace for the request is deleted after processing.

## 7. Prerequisites

Before setting up the project, ensure you have:

- Python installed.
- Git installed.
- A Google Gemini API key.
- Internet connectivity for downloading the embedding model and accessing the Gemini API.
- PowerPoint files in `.pptx` format for testing.

## 8. Installation and Setup

### Step 1: Clone the Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
```

Navigate to the project directory:

```bash
cd PresentationMaker
```

### Step 2: Create a Virtual Environment

```bash
python -m venv .venv
```

Activate the environment.

**Windows PowerShell:**

```powershell
.venv\Scripts\Activate.ps1
```

**Windows Command Prompt:**

```cmd
.venv\Scripts\activate.bat
```

### Step 3: Install Dependencies

Upgrade pip:

```bash
python -m pip install --upgrade pip
```

Install the project dependencies:

```bash
pip install -r requirements.txt
```

### Step 4: Configure the Gemini API Key

Create a `.env` file in the project root directory.

Add your Gemini API key:

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

The application uses the `GEMINI_API_KEY` environment variable to authenticate requests to the Gemini API.

**Important:** Never commit your actual `.env` file or expose your API key on GitHub.

### Step 5: Verify the Setup

If `check_setup.py` is available in your project, run:

```bash
python check_setup.py
```

This checks the Python environment, required packages, and MiniLM embedding model initialization.

The first model initialization downloads the required model files from Hugging Face. Subsequent runs can reuse the local cache.

## 9. Running the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

Streamlit will provide a local URL, typically:

```text
http://localhost:8501
```

Open the URL in your browser.

### Using PresentationMaker

1. Upload one or more `.pptx` files.
2. Enter the topic or describe the presentation you want to generate.
3. Specify the desired slide count (between 1 and 20).
4. Enter the target audience.
5. Enter the presentation objective.
6. Click **Generate Presentation**.
7. Wait for the extraction, embedding, retrieval, evidence selection, and generation stages to complete.
8. Download the generated PowerPoint presentation.

## 10. Running the Pipeline Without Streamlit

The individual pipeline scripts can also be executed from the terminal.

First, place the source PowerPoint files inside:

```text
input_ppts/
```

Run the scripts in the following order:

**1. Extract slide content**

```bash
python read_ppt.py
```

**2. Generate text embeddings**

```bash
python build_embeddings.py
```

**3. Extract image records**

```bash
python build_image_records.py
```

**4. Generate image embeddings**

```bash
python build_image_embeddings.py
```

**5. Generate the presentation**

```bash
python generate_presentation.py --query "Explain machine learning algorithms" --slides 10 --audience "Students" --objective "Explain the main concepts clearly"
```

The generated PowerPoint file is saved in the configured output directory.

## 11. Workspace Management

PresentationMaker uses separate workspaces for individual Streamlit generation requests.

Each request creates a temporary directory with the following structure:

```text
presentation_request_xxxxxx/
│
├── input_ppts/
│   └── uploaded_files.pptx
│
└── output/
    ├── slide_records.json
    ├── slide_embeddings.npy
    ├── image_records.json
    ├── image_embeddings.npy
    └── generated_presentation.pptx
```

The exact generated files depend on the processing stages.

The `PRESENTATIONMAKER_WORKSPACE` environment variable allows the pipeline scripts to use the request-specific workspace instead of the shared project directories.

This design provides:

- Isolation between generation requests
- No mixing of uploaded source files
- Request-specific records and embeddings
- Automatic cleanup of temporary processing data

The generated presentation is retained in memory for download before the temporary workspace is removed.

## 12. Configuration

The central configuration is maintained in `config.py`.

Important settings include:

| Setting | Description |
|---|---|
| `EMBEDDING_MODEL` | MiniLM model used for semantic embeddings |
| `EMBEDDING_DIM` | Embedding dimension (384) |
| `MAX_SEQ_LENGTH` | Maximum token sequence length (256) |
| `GEMINI_MODEL` | Gemini model used for content generation |
| `INPUT_DIR` | Input PowerPoint directory |
| `OUTPUT_DIR` | Processing and output directory |
| `DATA_DIR` | Shared project data directory |
| `PRESENTATIONMAKER_WORKSPACE` | Optional request-specific workspace path |

The currently configured Gemini model is `gemini-3.1-flash-lite`.

## 13. Current Implementation Notes and Limitations

- The current Streamlit pipeline processes uploaded `.pptx` files.
- OCR processing is temporarily disabled in the Streamlit pipeline. Text embedded inside images is therefore not currently extracted through the OCR stage.
- Image retrieval depends on the extracted image records and their associated metadata and embeddings.
- Gemini API access requires a valid API key and internet connectivity.
- The first MiniLM initialization requires downloading model files from Hugging Face.
- The quality of generated presentations depends on the relevance, completeness, and quality of the uploaded source material.
- The system is designed to ground generated content in retrieved source material rather than intentionally introduce unsupported information.

## 14. Security and Data Handling

- API credentials must be stored in `.env`.
- `.env`, virtual environments, uploaded PowerPoint files, generated outputs, and temporary workspaces should not be committed to Git.
- Streamlit generation requests use separate temporary workspaces.
- Temporary workspaces are removed after the generation process finishes.

## 15. Future Improvements

Potential areas for further development include:

- Reintroducing and evaluating OCR for text embedded in images.
- Improving image understanding and diagram retrieval.
- Improving retrieval using chunk-level indexing and reranking.
- Enhancing evidence coverage and presentation coherence.
- Improving slide layouts, themes, and visual consistency.
- Adding presentation preview and editing capabilities.
- Evaluating retrieval quality and generated presentation quality using systematic metrics.

---

## Author

**PresentationMaker**

An evolving project focused on combining information retrieval, semantic search, multimodal content processing, and LLM-based generation to automate the creation of source-grounded PowerPoint presentations.
