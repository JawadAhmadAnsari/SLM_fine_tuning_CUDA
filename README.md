# Enterprise SLM Fine-Tuning Framework on Nvidia RTX 3050 Laptop (CUDA)

This repository provides a production-ready framework for fine-tuning Small Language Models (SLMs) for specialized enterprise tasks. It is built for efficiency, reproducibility, and scalability, leveraging Unsloth for memory-optimized training and DagsHub for collaborative experiment tracking.

This project is compatible with **Google Colab** and standard local development environments, including Native Windows environments.

---

## Pipeline Architecture

The fine-tuning pipeline is designed to be configuration-driven, ensuring that experiments are reproducible and easy to manage without changing the source code. The entire lifecycle, encompassing training, evaluation, and drift detection, is orchestrated by `src/controller.py`.

1.  **Configuration Loading**: The process starts when the main training script (`scripts/train.py`) is executed. It calls the `train()` function in `src/train.py`, which begins by loading all hyperparameters from `configs/config.yaml` using the `get_config()` utility from `src/utils.py`.

2.  **Secret Management**: Sensitive credentials (like API tokens for DagsHub and Hugging Face) are loaded from a `.env` file at the project root using the `python-dotenv` library. This keeps secrets out of version control.

3.  **RAG (Retrieval Augmented Generation)**: The framework now includes a RAG module to provide context from local documents. `src/rag.py` uses `LangChain` and `ChromaDB` to index PDF and Excel documents from the `data/` directory into a knowledge base for context retrieval during inference.

4.  **Data Preparation**: The dataset specified in the `dataset.path` of the config is loaded from Hugging Face. The `src/data.py` module contains functions like `format_prompt` which transform the raw data into the structured input format required by the model for fine-tuning.

5.  **Model Loading**: The base model is loaded using `src/model.py`. This module uses Unsloth's `FastLanguageModel` for memory-optimized loading (e.g., in 4-bit precision). It then applies a LoRA (Low-Rank Adaptation) configuration, also defined in `config.yaml`, to prepare the model for efficient fine-tuning.

6.  **Training Execution**: The training process is managed by the `trl.SFTTrainer`. This trainer is configured with `transformers.TrainingArguments` and the LoRA-adapted model. All parameters for the trainer (batch size, learning rate, etc.) are pulled directly from the `training` section of `config.yaml`.

7.  **Experiment Tracking**: Throughout the training run, all parameters, metrics, and model artifacts are logged to MLflow. The MLflow tracking URI and credentials are automatically sourced from the `.env` file, seamlessly integrating with platforms like DagsHub.

---

## Self-Healing & Observability

This framework incorporates robust self-healing and observability features to ensure model reliability and performance in production:

* **Drift Detection**: The system continuously monitors for model degradation using `src/drift_detector.py`. It compares current model performance against a baseline and identifies significant drops in accuracy.

* **Automated Retraining**: Thresholds for triggering automated retraining are configurable in `configs/autoeval.yaml`. If accuracy degradation exceeds these predefined limits, the system can initiate a retraining workflow.

* **Comprehensive Observability with Langfuse**: All critical events, including model inputs, outputs, performance metrics, and drift detection decisions, are logged as full traces to [Langfuse](https://langfuse.com). This provides deep insights for debugging, performance monitoring, and understanding model behavior in real-time.

---

## Environment Setup

You can run this project either on Google Colab or in a Local Windows/Linux environment.

### Google Colab

1.  **Open Notebook**: Open `notebooks/colab_controller.ipynb` in Google Colab.
2.  **Configure Secrets**:
    * Click the **key icon (🔑)** in the left sidebar.
    * Add the following secrets: `MLFLOW_TRACKING_USERNAME`, `MLFLOW_TRACKING_PASSWORD`, `HUGGING_FACE_TOKEN`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST`.
    * Sign up for [Langfuse](https://langfuse.com) to get your API keys (`LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`) and ensure `LANGFUSE_HOST` is set to `https://cloud.langfuse.com`.
    * Ensure "Notebook access" is enabled for each.
    * The `MLFLOW_TRACKING_URI` can be set directly in the notebook.
3.  **Run Notebook**: Execute the cells sequentially. The notebook will mount your Drive, install all dependencies, and automatically handle data generation, training, and evaluation steps.

### Local Development (Native Windows & CUDA)

Running Unsloth and Triton natively on Windows requires specific C++ compilers and library path configurations. If you prefer to avoid this, we highly recommend using **WSL2 (Windows Subsystem for Linux)**.

If you are running natively on Windows, follow these steps strictly:

1.  **Install Microsoft C++ Build Tools**:
    * Download and install the **Build Tools for Visual Studio 2022**.
    * Ensure the **"Desktop development with C++"** workload is checked during installation.

2.  **Clone the Repository**:
    ```bash
    git clone <your-repo-url>
    cd <your-repo-name>
    ```

3.  **Launch the Correct Terminal**:
    * Do **not** use a standard Command Prompt or PowerShell. 
    * Open your Windows Start Menu and launch the **"x64 Native Tools Command Prompt for VS 2022"**. This ensures the `cl.exe` compiler is available in your PATH for Triton to compile kernels on the fly.
    * Navigate to your project directory in this prompt and type `code .` to launch VS Code with the correct environment variables inherited.

4.  **Create and Activate Virtual Environment**:
    ```cmd
    python -m venv .venv
    .venv\Scripts\activate
    ```

5.  **Fix Python Library Linking (Windows Specific)**:
    * Triton may fail to find your Python library if Python is installed in your `AppData` folder (Error: `fatal error LNK1104: cannot open file 'python311.lib'`).
    * Locate your `python311.lib` (or `python310.lib` depending on your version) in your main Python installation folder (e.g., `C:\Users\<User>\AppData\Local\Programs\Python\Python311\libs`).
    * Create a `libs` folder inside your project's virtual environment: `.venv\libs\`.
    * Copy the `python311.lib` file into this new `.venv\libs\` folder.

6.  **Install Dependencies**:
    * Install Unsloth (example for CUDA 12.1):
        ```cmd
        pip install "unsloth[cu121] @ git+[https://github.com/unslothai/unsloth.git](https://github.com/unslothai/unsloth.git)"
        ```
    * Install the remaining packages:
        ```cmd
        pip install -r requirements.txt
        ```

7.  **Patch Triton for CUDA 13.0 (If Applicable)**:
    * If you are using CUDA 13.0, Triton will throw a `RuntimeError: Triton only support CUDA 10.0 or higher`.
    * Open `.venv\Lib\site-packages\triton\backends\nvidia\compiler.py`.
    * Find the `ptx_get_version` function (around line 55).
    * Add the following check to bypass the error:
        ```python
        if major == 13:
            return 85
        ```

8.  **Configure Credentials**:
    * Create a `.env` file from the example: `copy .env.example .env`.
    * Edit `.env` to add your `MLFLOW_TRACKING_URI`, `MLFLOW_TRACKING_USERNAME`, `MLFLOW_TRACKING_PASSWORD`, and `HUGGING_FACE_TOKEN` (Ensure this is a **Write** access token).

---

## Configuration (`config.yaml`)

All aspects of the fine-tuning process are controlled by `configs/config.yaml`. Below is a breakdown of the key parameters.

| Section      | Parameter                       | Type    | Description                                                                                             |
|--------------|---------------------------------|---------|---------------------------------------------------------------------------------------------------------|
| **model** | `base_model`                    | String  | The identifier of the base model on Hugging Face (e.g., `unsloth/mistral-7b-v0.2-bnb-4bit`).            |
|              | `max_seq_length`                | Integer | The maximum sequence length for the model's context window.                                             |
|              | `load_in_4bit`                  | Boolean | If `True`, loads the model in 4-bit precision using Unsloth for significant memory savings.             |
| **lora** | `r`                             | Integer | The rank of the LoRA matrices. A higher rank means more trainable parameters.                           |
|              | `lora_alpha`                    | Integer | LoRA scaling factor. Often set to the same value as `r`.                                                |
|              | `target_modules`                | List    | A list of the model's internal modules (e.g., `q_proj`, `v_proj`) to apply LoRA adapters to.            |
|              | `lora_dropout`                  | Float   | Dropout probability for the LoRA layers to prevent overfitting.                                         |
|              | `use_rslora`                    | Boolean | If `True`, enables Rank-Stabilized LoRA, which can improve stability.                                   |
| **dataset** | `path`                          | String  | The Hugging Face path to the training dataset.                                                          |
| **training** | `per_device_train_batch_size`   | Integer | The batch size per GPU for training.                                                                    |
|              | `gradient_accumulation_steps` | Integer | Number of steps to accumulate gradients before performing a weight update.                              |
|              | `max_steps`                     | Integer | The total number of training steps to perform.                                                          |
|              | `learning_rate`                 | Float   | The initial learning rate for the optimizer.                                                            |
|              | `optim`                         | String  | The optimizer to use (e.g., `adamw_8bit` for a memory-efficient Adam optimizer).                        |
|              | `output_dir`                    | String  | The local directory where the final trained model adapters will be saved.                               |
|              | `push_to_hub`                   | Boolean | If `True`, automatically pushes the final model to the Hugging Face Hub repo specified in `deployment`. |
| **mlflow** | `experiment_name`               | String  | The name of the experiment under which runs will be logged in MLflow.                                   |
|              | `run_name`                      | String  | A specific name for the training run, used for easier identification in MLflow.                         |
| **deployment**| `hf_hub_repo`                  | String  | **CRITICAL:** The exact Hugging Face Hub repository ID (e.g., `YourUsername/your-model-name`). **You must create this empty repository on the Hugging Face website first** before running the script to avoid a 403 Forbidden Error. |


## How to Contribute

We welcome contributions from the team. Please follow these standard engineering guidelines to maintain code quality and a smooth workflow.

1.  **Create a Feature Branch**: All new work, whether a feature or a bug fix, should be done in a separate branch.
    ```bash
    git checkout -b feature/your-feature-name main
    ```

2.  **Follow Code Style**: Adhere to the existing code style (PEP 8 for Python). Run a linter if one is configured for the project.

3.  **Write Clear Commit Messages**: Write clear, concise, and descriptive commit messages that explain the *why* behind your changes.

4.  **Keep Pull Requests Focused**: A Pull Request (PR) should address a single, specific issue or feature. Avoid bundling unrelated changes into one PR.

5. **Review and Test**: Before submitting a PR, thoroughly test your changes. Ensure that the training script runs and that your changes have not introduced any regressions. Once submitted, another team member should review your PR.

## Standard Workflow

### Step 1: Generate Knowledge Base
Run the following script to index your local documents into a ChromaDB vector store.
```bash
python scripts/prepare_rag_data.py