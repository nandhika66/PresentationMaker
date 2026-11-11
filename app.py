
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import streamlit as st

from config import BASE_DIR


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="PresentationMaker",
    page_icon="📊",
    layout="wide"
)

st.title("PresentationMaker")
st.write("Generate presentations using your uploaded PowerPoint files.")


# --------------------------------------------------
# USER INPUTS
# --------------------------------------------------

st.header("1. Upload presentations")

uploaded_files = st.file_uploader(
    "Upload PPTX files",
    type=["pptx"],
    accept_multiple_files=True
)

st.header("2. Presentation requirements")

query = st.text_area(
    "What presentation would you like to create?",
    placeholder="Explain the machine learning algorithms..."
)

slide_count = st.number_input(
    "Number of slides",
    min_value=1,
    max_value=20,
    value=10
)

audience = st.text_input(
    "Target audience",
    value="General audience"
)

objective = st.text_input(
    "Presentation objective",
    value="Explain the requested topic clearly"
)


# --------------------------------------------------
# PIPELINE EXECUTION FUNCTION
# --------------------------------------------------

def run_script(script_name, workspace, extra_args=None):

    env = os.environ.copy()

    env["PRESENTATIONMAKER_WORKSPACE"] = str(workspace)

    command = [
        sys.executable,
        str(BASE_DIR / script_name)
    ]

    if extra_args:
        command.extend(extra_args)

    result = subprocess.run(
        command,
        cwd=BASE_DIR,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace"
    )

    if result.stdout:
        st.code(result.stdout[-12000:])

    if result.returncode != 0:
        error_message = result.stderr or "Unknown pipeline error"

        raise RuntimeError(
            f"{script_name} failed:\n{error_message[-6000:]}"
        )

    return result.stdout


# --------------------------------------------------
# GENERATION
# --------------------------------------------------

if st.button("Generate Presentation", type="primary"):

    if not uploaded_files:
        st.error("Please upload at least one PPTX file.")

    elif not query.strip():
        st.error("Please enter a presentation request.")

    else:

        workspace = None
        generated_data = None
        generated_name = None

        try:

            # Create a unique workspace for this request.
            workspace = Path(
                tempfile.mkdtemp(
                    prefix="presentation_request_",
                    dir=BASE_DIR
                )
            )

            input_dir = workspace / "input_ppts"
            output_dir = workspace / "output"

            input_dir.mkdir(parents=True)
            output_dir.mkdir(parents=True)

            # Save uploaded files.
            for index, uploaded_file in enumerate(uploaded_files):

                safe_name = Path(uploaded_file.name).name
                file_path = input_dir / f"{index}_{safe_name}"

                file_path.write_bytes(uploaded_file.getbuffer())

            st.success(
                f"{len(uploaded_files)} presentation(s) uploaded."
            )

            # Run each pipeline stage in order.
            pipeline_steps = [
                ("Extracting PowerPoint content", "read_ppt.py"),
                # OCR temporarily disabled.
                # Enable this later when evaluating OCR performance.
                #("Extracting text from images", "ocr_images.py"),
                ("Building text embeddings", "build_embeddings.py"),
                ("Extracting image records", "build_image_records.py"),
                ("Building image embeddings", "build_image_embeddings.py"),
            ]

            for step_name, script_name in pipeline_steps:

                with st.status(step_name, expanded=True) as status:

                    run_script(script_name, workspace)

                    status.update(
                        label=f"Completed: {step_name}",
                        state="complete"
                    )

            # Generate the final presentation.
            with st.status(
                "Generating your presentation",
                expanded=True
            ) as status:

                run_script(
                    "generate_presentation.py",
                    workspace,
                    [
                        "--query", query,
                        "--slides", str(int(slide_count)),
                        "--audience", audience,
                        "--objective", objective
                    ]
                )

                status.update(
                    label="Presentation generated successfully",
                    state="complete"
                )

            # Find the generated PowerPoint.
            ppt_files = list(output_dir.glob("*.pptx"))

            if not ppt_files:
                raise FileNotFoundError(
                    "The pipeline finished, but no PPTX file was found."
                )

            generated_file = max(
                ppt_files,
                key=lambda path: path.stat().st_mtime
            )

            # Read the generated PPT into memory BEFORE deleting workspace.
            generated_data = generated_file.read_bytes()
            generated_name = generated_file.name

            st.success("Your presentation is ready!")

        except Exception as error:

            st.error("Presentation generation failed.")
            st.exception(error)

        finally:

            # Delete this request's workspace.
            if workspace and workspace.exists():
                shutil.rmtree(workspace, ignore_errors=True)

        # The generated PPT remains available in memory.
        if generated_data:

            st.download_button(
                label="Download Presentation",
                data=generated_data,
                file_name=generated_name,
                mime=(
                    "application/vnd.openxmlformats-officedocument."
                    "presentationml.presentation"
                ),
                type="primary"
            )
