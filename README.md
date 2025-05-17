# Telepathy AI Desktop Automator
<a href="https://imgbb.com/"><img src="https://i.ibb.co/YCPDk1c/Screenshot-20250517-190343.png" alt="Screenshot-20250517-190343" border="0"></a>

Telepathy is an AI-powered desktop automation agent that observes your screen, understands context using multimodal AI (Ollama/Llava), and executes actions using PyAutoGUI. It features a Retrieval Augmented Generation (RAG) system that can leverage your Obsidian notes and learned command patterns to make more informed decisions.

## Features

*   **Visual UI Understanding:** Uses a multimodal LLM (e.g., Llava via Ollama) to interpret screenshots.
*   **Desktop Automation:** Controls mouse and keyboard via PyAutoGUI (clicks, typing, scrolling, hotkeys, etc.).
*   **Contextual Awareness:** Considers the active application and recent command history.
*   **Retrieval Augmented Generation (RAG):**
    *   **Obsidian Vault Integration:** Indexes your Markdown notes from an Obsidian vault into a ChromaDB vector store, allowing the AI to "consult" your notes for relevant information.
    *   **Learned Patterns:** Analyzes successful command sequences from history and stores them as "learned patterns" in ChromaDB to guide future actions.
*   **Command History:** Logs all attempted commands and their success status to an SQLite database.
*   **Extensible Actions:** Define custom high-level actions through an `ActionExtension` system.
*   **Configurable:** Uses a `.env` file for easy configuration of models, paths, and intervals.
*   **Failsafe:** Incorporates PyAutoGUI's failsafe (move mouse to a corner to stop).

## How it Works

1.  **Capture:** Periodically takes a screenshot of the current screen.
2.  **Contextualize:** Determines the foreground application and gathers recent command history.
3.  **Retrieve (RAG):**
    *   Queries ChromaDB for relevant snippets from your Obsidian notes based on the current context.
    *   Queries ChromaDB for "learned patterns" that match the current application and task.
4.  **Prompt AI:** Constructs a detailed prompt for the multimodal LLM, including:
    *   The current screenshot.
    *   Foreground application name.
    *   Screen resolution.
    *   Recent command history.
    *   Retrieved RAG context (Obsidian notes, learned patterns).
    *   A list of available actions (standard PyAutoGUI + custom extensions).
5.  **Generate Action:** The LLM analyzes the prompt and image, then generates a single command to execute.
6.  **Parse & Validate:** The LLM's response is parsed to extract the command and parameters, which are then validated (e.g., coordinates within screen bounds).
7.  **Execute:** The validated command is executed using PyAutoGUI.
8.  **Log:** The command, its context (application), and success status are logged to the SQLite history database.
9.  **Learn (Periodically/Startup):** The `LearningSystem` analyzes the command history to identify successful patterns, which are then embedded and stored in ChromaDB.
10. **Repeat:** The cycle continues.

## Prerequisites

1.  **Python 3.8+**
2.  **Ollama:**
    *   Install Ollama from [ollama.ai](https://ollama.ai/).
    *   Pull the required models:
        ```bash
        ollama pull llava:7b  # Or your chosen vision model
        ollama pull nomic-embed-text # Or your chosen embedding model
        ```
        Ensure these match the `OLLAMA_VISION_MODEL` and `OLLAMA_EMBED_MODEL` in your `.env` file.
3.  **Platform-specific dependencies for `get_foreground_app`:**
    *   **Linux:** `xdotool` and `ps` (from `procps` or `procps-ng`).
        ```bash
        sudo apt update
        sudo apt install xdotool procps
        ```
    *   **Windows:** `pywin32` is installed as part of `requirements.txt`.
    *   **macOS:** No specific extra tools are used by default, relies on PyAutoGUI's `getActiveWindow()`.
4.  **(Optional) Obsidian:** If you want to use Obsidian note indexing, ensure you have an Obsidian vault.

## Setup & Installation

1.  **Clone the repository (if applicable):**
    ```bash
    # git clone <your-repo-url>
    # cd <your-repo-name>
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Python dependencies:**
    The dependencies are listed in `code.txt`.  or use it directly:
    ```bash
    # mv code.txt 
    # pip install -r code.txt

    # Option 2: Install directly from code.txt
    pip install -r code.txt
    ```
    This will install `pyautogui`, `ollama`, `requests`, `python-dotenv`, `chromadb`, `psutil`, etc.

4.  **Create a `.env` file:**
    Copy the example or create a new `.env` file in the root directory:
    ```env
    # .env

    # Ollama Configuration
    OLLAMA_HOST=http://localhost:11434
    OLLAMA_VISION_MODEL=llava:7b
    OLLAMA_EMBED_MODEL=nomic-embed-text

    # Automation Engine Settings
    SCREENSHOT_INTERVAL=2 # Seconds between screenshots
    # FAILSAFE_CORNER is hardcoded to topLeft in the script for PyAutoGUI.FAILSAFEPOINT

    # ChromaDB (Vector Store for RAG)
    CHROMA_DB_PATH=./chroma_db_store # Path for ChromaDB persistence

    # Obsidian Integration (Optional)
    # Set this to the root path of your Obsidian vault if you want to index your notes
    OBSIDIAN_VAULT_PATH=/path/to/your/obsidian/vault
    # Example: OBSIDIAN_VAULT_PATH=C:/Users/YourUser/Documents/ObsidianVault
    # Example: OBSIDIAN_VAULT_PATH=/home/youruser/ObsidianVault
    ```

5.  **Verify Ollama:**
    Ensure Ollama is running and the models specified in `.env` are downloaded. You can test by visiting `http://localhost:11434` in your browser or using `ollama list` in your terminal.

## Configuration

Modify the `.env` file to customize:

*   `OLLAMA_HOST`: URL of your Ollama server.
*   `OLLAMA_VISION_MODEL`: The multimodal model Ollama should use for analyzing screenshots (e.g., `llava:7b`, `llava:13b`).
*   `OLLAMA_EMBED_MODEL`: The text embedding model Ollama should use for RAG (e.g., `nomic-embed-text`, `mxbai-embed-large`).
*   `SCREENSHOT_INTERVAL`: How often (in seconds) the agent takes a screenshot and attempts an action.
*   `CHROMA_DB_PATH`: Directory where ChromaDB will store its persistent data (vector embeddings).
*   `OBSIDIAN_VAULT_PATH`: (Optional) Absolute path to your Obsidian vault. If set, notes will be indexed on startup.

## Running the Application

Once configured, run the main script:

```bash
python code.py

MADE by Fiberr.co non-profit.
