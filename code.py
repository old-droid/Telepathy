import os
import time
import logging
import pyautogui
import ollama
import requests
import sqlite3
import glob
import re
import threading
import json # For metadata in ChromaDB
import chromadb
from chromadb.utils import embedding_functions
from requests.exceptions import StreamConsumedError
from dotenv import load_dotenv
from datetime import datetime
from typing import List, Dict, Any, Optional
from collections import deque
from abc import ABC, abstractmethod

# --- Configuration ---
load_dotenv()
OLLAMA_HOST = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
# Multimodal model for vision and command generation
OLLAMA_VISION_MODEL = os.getenv('OLLAMA_VISION_MODEL', 'llava:7b')
# Model for generating text embeddings (ensure this model is pulled in Ollama)
OLLAMA_EMBED_MODEL = os.getenv('OLLAMA_EMBED_MODEL', 'nomic-embed-text')

SCREENSHOT_INTERVAL = int(os.getenv('SCREENSHOT_INTERVAL', '2')) # How often to take a screenshot (seconds)
FAILSAFE_CORNER = 'topLeft' # PyAutoGUI failsafe corner
MAX_PROCESSED_COMMANDS = 20 # How many recent commands to remember for prompt context
CHROMA_DB_PATH = os.getenv('CHROMA_DB_PATH', './chroma_db_store') # Path for ChromaDB persistence
OBSIDIAN_VAULT_PATH = os.getenv('OBSIDIAN_VAULT_PATH')

# Collection names for ChromaDB
CHROMA_COLLECTION_OBSIDIAN = "obsidian_notes"
CHROMA_COLLECTION_LEARNED_PATTERNS = "learned_patterns"

# --- Logging Setup ---
def setup_logger(name, level=logging.DEBUG, file_name='automation.log'):
    logger = logging.getLogger(name)
    if logger.hasHandlers(): # Prevent duplicate handlers
        logger.handlers.clear()

    logger.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Stream Handler (console)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # File Handler
    file_handler = logging.FileHandler(file_name, mode='w')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger

logger = setup_logger('telepathy')
rag_logger = setup_logger('rag', file_name='rag_automation.log') # Separate log for RAG specifics

# --- ChromaDB Manager ---
class ChromaDBManager:
    def __init__(self, path: str, ollama_host: str, embed_model_name: str):
        self.client = chromadb.PersistentClient(path=path)
        self.ollama_host = ollama_host
        self.embed_model_name = embed_model_name
        self.ollama_ef = embedding_functions.OllamaEmbeddingFunction(
            url=f"{self.ollama_host}/api/embeddings", # Corrected API endpoint
            model_name=self.embed_model_name,
        )
        logger.info(f"ChromaDB client initialized at path: {path}")
        logger.info(f"ChromaDB using Ollama embeddings from host: {ollama_host}, model: {embed_model_name}")
        self._initialize_collections()

    def _initialize_collections(self):
        try:
            self.get_or_create_collection(CHROMA_COLLECTION_OBSIDIAN)
            self.get_or_create_collection(CHROMA_COLLECTION_LEARNED_PATTERNS)
            logger.info("ChromaDB collections initialized.")
        except Exception as e:
            logger.error(f"Error initializing ChromaDB collections: {e}", exc_info=True)


    def get_or_create_collection(self, name: str):
        try:
            return self.client.get_or_create_collection(name=name, embedding_function=self.ollama_ef)
        except Exception as e:
            logger.error(f"Error getting or creating collection '{name}': {e}", exc_info=True)
            raise

    def add_documents(self, collection_name: str, documents: List[str], metadatas: List[Dict], ids: List[str]):
        if not documents:
            return
        try:
            collection = self.client.get_collection(name=collection_name, embedding_function=self.ollama_ef)
            collection.add(documents=documents, metadatas=metadatas, ids=ids)
            rag_logger.debug(f"Added {len(documents)} documents to collection '{collection_name}'.")
        except Exception as e:
            rag_logger.error(f"Error adding documents to ChromaDB collection '{collection_name}': {e}", exc_info=True)

    def query_collection(self, collection_name: str, query_texts: List[str], n_results: int = 3) -> List[Dict[str, Any]]:
        try:
            collection = self.client.get_collection(name=collection_name, embedding_function=self.ollama_ef)
            results = collection.query(query_texts=query_texts, n_results=n_results, include=['documents', 'metadatas', 'distances'])

            # Flatten results and format them
            formatted_results = []
            if results and results.get('ids'): # Check if 'ids' (and thus other lists) exist
                for i in range(len(results['ids'][0])): # Iterate through results for the first (and only) query_text
                    doc_id = results['ids'][0][i]
                    document = results['documents'][0][i] if results['documents'] and results['documents'][0] else None
                    metadata = results['metadatas'][0][i] if results['metadatas'] and results['metadatas'][0] else {}
                    distance = results['distances'][0][i] if results['distances'] and results['distances'][0] else float('inf')
                    if document: # Only include if document text is present
                        formatted_results.append({
                            "id": doc_id,
                            "document": document,
                            "metadata": metadata,
                            "distance": distance
                        })
            rag_logger.debug(f"ChromaDB query to '{collection_name}' for '{query_texts}' returned {len(formatted_results)} results.")
            return formatted_results
        except Exception as e:
            rag_logger.error(f"Error querying ChromaDB collection '{collection_name}': {e}", exc_info=True)
            return []

# --- History Database (SQLite for command logs) ---
class HistoryDatabase:
    def __init__(self, db_path='history.db'):
        self.conn = None
        try:
            self.conn = sqlite3.connect(db_path)
            self._create_tables()
            logger.info(f"Successfully connected to history database: {db_path}")
        except sqlite3.Error as e:
            logger.critical(f"History database connection error: {str(e)}")

    def _create_tables(self):
        if not self.conn: return
        try:
            self.conn.execute('''CREATE TABLE IF NOT EXISTS command_history
                (id INTEGER PRIMARY KEY AUTOINCREMENT,
                 command TEXT NOT NULL,
                 app_context TEXT,
                 success INTEGER,
                 timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
            self.conn.commit()
            logger.debug("History database tables checked/created.")
        except sqlite3.Error as e:
            logger.error(f"Error creating history database tables: {str(e)}")

    def log_command(self, command: str, app_context: str, success: bool):
        if not self.conn: return
        try:
            self.conn.execute('INSERT INTO command_history (command, app_context, success) VALUES (?, ?, ?)',
                            (command, app_context, int(success)))
            self.conn.commit()
            logger.debug(f"Logged command to history: '{command}' (App: {app_context}, Success: {success})")
        except sqlite3.Error as e:
            logger.error(f"Error logging command to history: {str(e)}")

    def get_recent_successful_commands(self, app_context: Optional[str] = None, limit: int = 10) -> List[Dict]:
        if not self.conn: return []
        cursor = self.conn.cursor()
        try:
            query = '''SELECT command, app_context, timestamp FROM command_history
                       WHERE success = 1 '''
            params = []
            if app_context:
                query += 'AND app_context LIKE ? '
                params.append(f'%{app_context}%')
            query += 'ORDER BY timestamp DESC LIMIT ?'
            params.append(limit)

            cursor.execute(query, tuple(params))
            return [{'command': row[0], 'app_context': row[1], 'timestamp': row[2]} for row in cursor.fetchall()]
        except sqlite3.Error as e:
            logger.error(f"Error retrieving recent successful commands: {str(e)}")
            return []

    def close(self):
        if self.conn:
            self.conn.close()
            logger.info("History database connection closed.")

# --- Obsidian Indexer ---
class ObsidianIndexer:
    def __init__(self, vault_path: Optional[str], chroma_manager: ChromaDBManager):
        self.vault_path = vault_path
        self.chroma_manager = chroma_manager
        if self.vault_path and not os.path.exists(self.vault_path):
             logger.warning(f"Obsidian vault path not found: {self.vault_path}. Indexing will be skipped.")
             self.vault_path = None

    def index_notes(self):
        if not self.vault_path:
            logger.info("Obsidian indexing skipped (no valid vault path).")
            return

        logger.info(f"Starting Obsidian indexing for vault: {self.vault_path}")
        md_files = glob.glob(f'{self.vault_path}/**/*.md', recursive=True)
        
        documents, metadatas, ids = [], [], []
        batch_size = 50 # Process in batches to avoid overwhelming embedding model or DB

        for i, file_path in enumerate(md_files):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Simple unique ID: hash of file path
                doc_id = f"obsidian_{os.path.normpath(file_path).replace(os.sep, '_')}"
                doc_id = re.sub(r'[^a-zA-Z0-9_-]', '', doc_id) # Clean ID for Chroma

                documents.append(content)
                metadatas.append({'source': file_path, 'type': 'obsidian_note', 'indexed_at': datetime.now().isoformat()})
                ids.append(doc_id)

                if (i + 1) % batch_size == 0 or (i + 1) == len(md_files):
                    self.chroma_manager.add_documents(CHROMA_COLLECTION_OBSIDIAN, documents, metadatas, ids)
                    rag_logger.info(f"Indexed batch of {len(documents)} Obsidian notes.")
                    documents, metadatas, ids = [], [], [] # Reset for next batch

            except Exception as e:
                rag_logger.error(f'Error indexing Obsidian note {file_path}: {str(e)}')
        
        logger.info(f"Finished Obsidian indexing. Processed {len(md_files)} markdown files.")

# --- Learning System ---
class LearningSystem:
    def __init__(self, history_db: HistoryDatabase, chroma_manager: ChromaDBManager):
        self.history_db = history_db
        self.chroma_manager = chroma_manager

    def analyze_and_learn_patterns(self):
        """Analyzes successful command history and stores contextual patterns in ChromaDB."""
        logger.info("Learning System: Analyzing command history for patterns...")
        # Get recent successful commands, possibly grouped by app context
        # For simplicity, let's analyze commands associated with specific apps
        
        # Example: Get all app contexts from history
        cursor = self.history_db.conn.cursor()
        cursor.execute("SELECT DISTINCT app_context FROM command_history WHERE success = 1 AND app_context IS NOT NULL")
        app_contexts = [row[0] for row in cursor.fetchall() if row[0] and row[0] not in ["Unknown (Error)", "Unknown (Win32)", "Unknown (Linux - lib missing)"]]

        learned_docs, learned_metadatas, learned_ids = [], [], []
        
        for app_name in app_contexts:
            if not app_name or app_name.lower() == "unknown": continue
            
            successful_commands = self.history_db.get_recent_successful_commands(app_context=app_name, limit=20)
            if len(successful_commands) > 2: # Need at least a few commands to form a pattern
                # Create a textual representation of this pattern
                pattern_description = f"In application '{app_name}', the following sequence of commands was successful:\n"
                for cmd_info in reversed(successful_commands[:5]): # Take up to 5 most recent for a concise pattern
                    pattern_description += f"- {cmd_info['command']} (at {cmd_info['timestamp']})\n"
                pattern_description += "This might be a useful strategy or common workflow."
                
                doc_id = f"pattern_{app_name.replace('.', '_').lower()}_{int(time.time())}"
                doc_id = re.sub(r'[^a-zA-Z0-9_-]', '', doc_id)

                learned_docs.append(pattern_description)
                learned_metadatas.append({
                    'source': 'learning_system',
                    'app_context': app_name,
                    'type': 'learned_pattern',
                    'timestamp': datetime.now().isoformat()
                })
                learned_ids.append(doc_id)

        if learned_docs:
            self.chroma_manager.add_documents(CHROMA_COLLECTION_LEARNED_PATTERNS, learned_docs, learned_metadatas, learned_ids)
            rag_logger.info(f"Learning System: Stored {len(learned_docs)} new patterns in ChromaDB.")
        else:
            logger.info("Learning System: No new significant patterns found to store.")

# --- Action Extension System ---
class ActionExtension(ABC):
    @abstractmethod
    def get_command_name(self) -> str:
        """Returns the unique command name for this extension (e.g., MY_CUSTOM_ACTION)."""
        pass

    @abstractmethod
    def get_command_description(self) -> str:
        """Returns a brief description of the command and its parameters for the LLM prompt."""
        pass

    @abstractmethod
    def execute(self, params_str: str, engine: 'AutomationEngine') -> bool:
        """
        Executes the custom action.
        :param params_str: The string of parameters provided by the LLM.
        :param engine: Instance of the AutomationEngine for context if needed.
        :return: True if successful, False otherwise.
        """
        pass

# --- Core Automation Engine ---
class AutomationEngine:
    def __init__(self, history_db: HistoryDatabase, chroma_manager: ChromaDBManager, ollama_client):
        self.running = threading.Event()
        self.screenshot_interval = SCREENSHOT_INTERVAL
        self.processed_commands = deque(maxlen=MAX_PROCESSED_COMMANDS)
        self.history_db = history_db
        self.chroma_manager = chroma_manager
        self.ollama_client = ollama_client # ollama.Client instance
        self.extensions: Dict[str, ActionExtension] = {}

        pyautogui.FAILSAFE = True
        pyautogui.PAUSE = 0.1 # Small default pause for stability
        pyautogui.FAILSAFEPOINT = pyautogui.Point(0, 0) # Default top-left

    def register_extension(self, extension: ActionExtension):
        name = extension.get_command_name().upper()
        if name in self.extensions:
            logger.warning(f"Extension command '{name}' already registered. Overwriting.")
        self.extensions[name] = extension
        logger.info(f"Registered action extension: {name}")

    def take_screenshot(self) -> Optional[str]:
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            screenshot_dir = "screenshots"
            os.makedirs(screenshot_dir, exist_ok=True)
            screenshot_path = os.path.join(screenshot_dir, f"ss_{timestamp}.png")
            pyautogui.screenshot(screenshot_path)
            logger.debug(f"Screenshot captured: {screenshot_path}")
            return screenshot_path
        except Exception as e:
            logger.error(f"Failed to capture screenshot: {str(e)}")
            return None

    def get_foreground_app(self) -> str:
        try:
            import sys
            if sys.platform == "win32":
                import ctypes
                from ctypes import wintypes
                hwnd = ctypes.windll.user32.GetForegroundWindow()
                pid = wintypes.DWORD()
                ctypes.windll.user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
                PROCESS_QUERY_INFORMATION = 0x0400
                h_process = ctypes.windll.kernel32.OpenProcess(PROCESS_QUERY_INFORMATION, False, pid)
                if h_process:
                    exe_path = (ctypes.c_char * wintypes.MAX_PATH)() # Use MAX_PATH
                    if ctypes.windll.psapi.GetModuleFileNameExA(h_process, None, ctypes.byref(exe_path), ctypes.sizeof(exe_path)) > 0:
                        app_name = os.path.basename(exe_path.value.decode(errors='ignore'))
                        ctypes.windll.kernel32.CloseHandle(h_process)
                        return app_name
                    ctypes.windll.kernel32.CloseHandle(h_process)
                length = ctypes.windll.user32.GetWindowTextLengthA(hwnd)
                buff = ctypes.create_string_buffer(length + 1)
                ctypes.windll.user32.GetWindowTextA(hwnd, buff, length + 1)
                return buff.value.decode(errors='ignore') if buff.value else "Unknown (Win32)"
            elif sys.platform.startswith("linux"):
                 try:
                     import subprocess
                     active_window_id = subprocess.check_output(["xdotool", "getactivewindow"]).decode("utf-8").strip()
                     # Get PID from window ID
                     pid_output = subprocess.check_output(["xdotool", "getwindowpid", active_window_id]).decode("utf-8").strip()
                     if pid_output.isdigit():
                         pid = pid_output
                         # Get command from PID
                         process_name = subprocess.check_output(["ps", "-p", pid, "-o", "comm="]).decode("utf-8").strip()
                         return process_name
                     else: # Fallback to window name if PID method fails
                        window_name = subprocess.check_output(["xdotool", "getwindowname", active_window_id]).decode("utf-8").strip()
                        return window_name if window_name else "Unknown (Linux)"
                 except (FileNotFoundError, subprocess.CalledProcessError) as e:
                     logger.warning("Install xdotool and ensure ps command is available for Linux foreground app detection: " + str(e))
                     return "Unknown (Linux - lib missing)"
            else: # Fallback for other OS or if above fails
                active_window = pyautogui.getActiveWindow()
                return active_window.title if active_window else f"Unknown ({sys.platform})"
        except Exception as e:
            logger.error(f"Failed to get foreground app: {str(e)}", exc_info=True)
            return "Unknown (Error)"


    def process_image_with_ai(self, screenshot_path: str) -> List[str]:
        if not screenshot_path: return []

        try:
            current_app = self.get_foreground_app()
            screen_width, screen_height = pyautogui.size()

            # --- RAG: Retrieve relevant context from ChromaDB ---
            rag_query_text = f"Current application: {current_app}. Recent actions: {', '.join(list(self.processed_commands)[-5:]) if self.processed_commands else 'None'}."
            
            obsidian_contexts = self.chroma_manager.query_collection(
                CHROMA_COLLECTION_OBSIDIAN, [rag_query_text], n_results=2
            )
            learned_pattern_contexts = self.chroma_manager.query_collection(
                CHROMA_COLLECTION_LEARNED_PATTERNS, [rag_query_text], n_results=1
            )

            rag_context_str = "\n\n--- Relevant Knowledge ---\n"
            if not obsidian_contexts and not learned_pattern_contexts:
                rag_context_str += "No specific relevant knowledge found in your notes or learned patterns.\n"
            
            if obsidian_contexts:
                rag_context_str += "From your Obsidian notes:\n"
                for ctx in obsidian_contexts:
                    rag_context_str += f"- Source: {ctx['metadata'].get('source', 'N/A')}, Content Snippet: {ctx['document'][:300]}...\n"
            
            if learned_pattern_contexts:
                rag_context_str += "From learned patterns:\n"
                for ctx in learned_pattern_contexts:
                    rag_context_str += f"- Pattern: {ctx['document'][:400]}...\n"
            rag_context_str += "--- End Relevant Knowledge ---\n"
            
            # --- Build Extension Command List for Prompt ---
            extension_command_docs = ""
            if self.extensions:
                extension_command_docs = "\nAvailable Custom Actions (Extensions):\n"
                for i, (name, ext) in enumerate(self.extensions.items()):
                    extension_command_docs += f"{i+11}. {name}: {ext.get_command_description()}\n" # Start numbering after built-ins

            # --- Prompt Template ---
            prompt_template = f'''
You are an AI controlling a computer using pyautogui commands based on a screenshot of the active window.
Your task is to analyze the visual information and context to decide the next best action.
If a text editor is active, you might write a short poem or a note. Otherwise, interact with the UI appropriately.

Generate commands using the following strict format only:
COMMAND: ACTION PARAMETERS

Available Standard Actions and Required Parameters:
1.  CLICK X,Y: Integer pixel coordinates. Example: COMMAND: CLICK 1234,567
2.  DOUBLECLICK X,Y: Integer pixel coordinates. Example: COMMAND: DOUBLECLICK 123,456
3.  RIGHTCLICK X,Y: Integer pixel coordinates. Example: COMMAND: RIGHTCLICK 123,456
4.  MOVETO X,Y: Integer pixel coordinates. Example: COMMAND: MOVETO 789,101
5.  TYPE "text to type" {{optional_keys}}: Text in double quotes. Special keys (enter, tab, etc.) in curly braces, comma-separated. Example: COMMAND: TYPE "Hello" {{enter}}
6.  HOTKEY key1+key2: Keys joined by '+'. Example: COMMAND: HOTKEY ctrl+s
7.  SCROLL amount: Integer (positive for down, negative for up). Example: COMMAND: SCROLL -200
8.  DRAG X1,Y1 X2,Y2: Start and end coordinates. Example: COMMAND: DRAG 100,100 200,200
9.  KEYDOWN key: Single key name. Example: COMMAND: KEYDOWN shift
10. KEYUP key: Single key name. Example: COMMAND: KEYUP shift
{extension_command_docs}

Validation Rules:
- Coordinates (X,Y) must be within screen resolution: {screen_width}x{screen_height}.
- Text for TYPE must be in double quotes. Special keys in {{}}.
- Only generate ONE command per response.
- Focus on a single, logical step. Avoid recently processed commands if UI is unchanged.

Application Context:
- Foreground App: {current_app}
- Screen Resolution: {screen_width}x{screen_height}
- Recent Commands (last {len(self.processed_commands)}): {list(self.processed_commands)}
{rag_context_str}
Based on the current UI, context, and knowledge, generate the *single most appropriate* COMMAND:
'''
            # --- End Prompt Template ---

            logger.debug(f"Sending prompt to Ollama vision model: {OLLAMA_VISION_MODEL}")
            response = self.ollama_client.generate(
                model=OLLAMA_VISION_MODEL,
                prompt=prompt_template,
                images=[screenshot_path],
                stream=False # Simpler to handle non-streamed for single command
            )
            raw_response = response.get('response', '').strip()
            logger.debug(f"Raw AI response:\n---\n{raw_response}\n---")

            return self.parse_commands(raw_response, screen_width, screen_height)

        except requests.exceptions.ConnectionError:
            logger.error(f"Connection error with Ollama ({self.ollama_client.host}). Is Ollama running?")
            return []
        except Exception as e:
            logger.error(f"AI processing error: {str(e)}", exc_info=True)
            return []
        finally:
            if screenshot_path and os.path.exists(screenshot_path):
                try:
                    os.remove(screenshot_path)
                except Exception as e:
                    logger.warning(f"Failed to delete screenshot {screenshot_path}: {str(e)}")

    def parse_commands(self, response_text: str, screen_width: int, screen_height: int) -> List[str]:
        valid_commands = []
        # Regex for TYPE: "text content"{optional,keys} or "text content"
        type_regex = re.compile(r'^"([^"]*)"(?:\s*\{([^}]+)\})?$')
        # Regex for standard commands and extensions
        command_line_regex = re.compile(r'^COMMAND:\s*([A-Z_0-9]+)\s*(.*)', re.IGNORECASE)

        for line in response_text.split('\n'):
            line = line.strip()
            match = command_line_regex.match(line)
            if not match:
                continue

            action = match.group(1).upper()
            params = match.group(2).strip()
            full_command_for_history = f"{action} {params}" # Store the full command form

            logger.debug(f"Parsing - Action: '{action}', Raw Params: '{params}'")

            valid = False
            if action in ['CLICK', 'DOUBLECLICK', 'RIGHTCLICK', 'MOVETO']:
                try:
                    x_str, y_str = params.split(',', 1)
                    x, y = int(x_str.strip()), int(y_str.strip())
                    if 0 <= x < screen_width and 0 <= y < screen_height:
                        valid_commands.append(f'{action} {x},{y}')
                        valid = True
                    else:
                        logger.warning(f"Invalid {action}: Coords ({x},{y}) out of bounds {screen_width}x{screen_height}")
                except: logger.warning(f"Invalid {action} params: '{params}'")
            
            elif action == 'TYPE':
                type_match = type_regex.match(params)
                if type_match:
                    text_content = type_match.group(1) # Already unquoted by regex
                    special_keys_str = type_match.group(2)
                    
                    cmd_str = f'TYPE "{text_content.replace("\"", "\\\"")}"' # Re-add quotes for internal consistency
                    if special_keys_str:
                        keys = [k.strip().lower() for k in special_keys_str.split(',') if k.strip()]
                        if keys: cmd_str += f' {{{",".join(keys)}}}'
                    valid_commands.append(cmd_str)
                    valid = True
                else: logger.warning(f"Invalid TYPE params format: '{params}'")

            elif action == 'HOTKEY':
                keys = [k.strip().lower() for k in params.split('+') if k.strip()]
                if keys:
                    valid_commands.append(f'HOTKEY {"+".join(keys)}')
                    valid = True
                else: logger.warning(f"Invalid HOTKEY params: '{params}'")

            elif action == 'SCROLL':
                try:
                    amount = int(params.strip())
                    valid_commands.append(f'SCROLL {amount}')
                    valid = True
                except: logger.warning(f"Invalid SCROLL param: '{params}'")

            elif action == 'DRAG':
                try:
                    # Expects X1,Y1 X2,Y2 (potentially with or without comma)
                    coords = re.findall(r'\d+', params)
                    if len(coords) == 4:
                        x1, y1, x2, y2 = map(int, coords)
                        if all(0 <= c < (screen_width if i % 2 == 0 else screen_height) for i, c in enumerate([x1,y1,x2,y2])):
                            valid_commands.append(f'DRAG {x1},{y1} {x2},{y2}')
                            valid = True
                        else: logger.warning(f"Invalid DRAG: Coords out of bounds")
                    else: logger.warning(f"Invalid DRAG params format: '{params}' (expected 4 numbers)")
                except: logger.warning(f"Invalid DRAG params: '{params}'")
            
            elif action in ['KEYDOWN', 'KEYUP']:
                key = params.strip().lower()
                if key: # Basic check, pyautogui will validate further
                    valid_commands.append(f'{action} {key}')
                    valid = True
                else: logger.warning(f"Invalid {action} param: '{params}' (key name missing)")
            
            elif action in self.extensions: # Handle registered extensions
                # For extensions, we pass the raw param string; the extension handles parsing
                valid_commands.append(f'{action} {params}')
                valid = True
            
            else:
                logger.warning(f"Unrecognized command action '{action}' in line: '{line}'")

            if valid:
                 self.processed_commands.append(full_command_for_history) # Add successfully parsed command form to history
                 logger.debug(f"Parsed valid command: {valid_commands[-1]}")

        if not valid_commands and response_text: # If AI responded but no valid command parsed
             logger.warning(f"AI response contained text but no valid command was parsed: '{response_text}'")


        logger.info(f"Finished parsing AI response. Found {len(valid_commands)} valid command(s).")
        return valid_commands # Expecting only one command as per prompt

    def execute_command(self, command: str):
        success = False
        current_app_for_log = self.get_foreground_app() # Get app context before execution
        try:
            logger.info(f"Executing command: {command}")
            parts = command.split(' ', 1)
            action = parts[0].upper()
            params = parts[1] if len(parts) > 1 else ''

            if action in ['CLICK', 'DOUBLECLICK', 'RIGHTCLICK', 'MOVETO']:
                x_str, y_str = params.split(',', 1)
                x, y = int(x_str), int(y_str)
                if action != 'MOVETO': pyautogui.moveTo(x, y, duration=0.1) # Move before action
                
                if action == 'CLICK': pyautogui.click(x, y)
                elif action == 'DOUBLECLICK': pyautogui.doubleClick(x, y)
                elif action == 'RIGHTCLICK': pyautogui.rightClick(x, y)
                elif action == 'MOVETO': pyautogui.moveTo(x, y, duration=0.2)
                success = True
            
            elif action == 'TYPE':
                type_regex_exec = re.compile(r'^"([^"]*)"(?:\s*\{([^}]+)\})?$') # Match "text" {keys} or "text"
                match = type_regex_exec.match(params)
                if match:
                    text_content = match.group(1)
                    special_keys_str = match.group(2)
                    
                    pyautogui.typewrite(text_content, interval=0.01) # Use typewrite for better special char handling
                    if special_keys_str:
                        keys_to_press = [k.strip().lower() for k in special_keys_str.split(',') if k.strip()]
                        for key in keys_to_press:
                            pyautogui.press(key)
                            time.sleep(0.05) # Small delay after pressing special keys
                    success = True
                else: logger.error(f"Execution failed: Malformed TYPE params '{params}'")

            elif action == 'HOTKEY':
                keys = [k.strip() for k in params.split('+')]
                pyautogui.hotkey(*keys)
                success = True

            elif action == 'SCROLL':
                amount = int(params)
                pyautogui.scroll(amount)
                success = True
            
            elif action == 'DRAG':
                coords_str = params.replace(',', ' ').split() # "X1,Y1 X2,Y2" or "X1 Y1 X2 Y2"
                x1, y1, x2, y2 = map(int, coords_str)
                pyautogui.moveTo(x1, y1, duration=0.1)
                pyautogui.dragTo(x2, y2, duration=0.3)
                success = True

            elif action in ['KEYDOWN', 'KEYUP']:
                key = params.strip()
                if action == 'KEYDOWN': pyautogui.keyDown(key)
                elif action == 'KEYUP': pyautogui.keyUp(key)
                success = True
            
            elif action in self.extensions:
                extension_instance = self.extensions[action]
                logger.info(f"Executing extension command: {action} with params: '{params}'")
                success = extension_instance.execute(params, self)
            
            else:
                logger.error(f"Execution failed: Unknown action '{action}' for command: {command}")

        except pyautogui.FailSafeException:
            logger.critical(f"FAILSAFE TRIGGERED during execution of '{command}'! Automation stopping.")
            self.stop()
            success = False # Failsafe means command did not complete successfully
        except Exception as e:
            logger.error(f"Execution failed for command '{command}': {str(e)}", exc_info=True)
            success = False
        finally:
            self.history_db.log_command(command, current_app_for_log, success)
            if success:
                logger.info(f"Successfully executed: {command}")
            else:
                logger.warning(f"Failed to execute or error during: {command}")
            time.sleep(0.5) # Small pause after any command execution

    def run_loop(self):
        self.running.set()
        logger.info(f"Starting AI-Automation loop. Interval: {self.screenshot_interval}s, Vision Model: {OLLAMA_VISION_MODEL}")
        logger.info(f"Failsafe: Move mouse to {FAILSAFE_CORNER} corner (top-left) to stop.")

        while self.running.is_set():
            try:
                start_time = time.time()
                ss_path = self.take_screenshot()
                if not ss_path:
                    logger.error("Failed to get screenshot, skipping iteration.")
                    time.sleep(self.screenshot_interval) # Still respect interval
                    continue

                ai_commands = self.process_image_with_ai(ss_path)

                if ai_commands: # Prompt asks for ONE command
                    command_to_execute = ai_commands[0]
                    # Simple check: don't execute the exact same command if it was the very last one and likely failed or did nothing
                    if self.processed_commands and command_to_execute == self.processed_commands[-1]:
                         # A more sophisticated check would involve analyzing if UI actually changed
                         logger.info(f"Skipping re-execution of identical last command: {command_to_execute}")
                    else:
                        self.execute_command(command_to_execute)
                        # self.processed_commands.append(command_to_execute) # Moved to parse_commands upon successful parsing
                else:
                    logger.info("No command generated by AI for the current state.")

            except KeyboardInterrupt:
                logger.info("Keyboard interrupt detected, stopping loop.")
                self.stop()
            except pyautogui.FailSafeException:
                 logger.critical("FAILSAFE TRIGGERED in main loop! Automation stopping.")
                 self.stop()
            except Exception as e:
                logger.error(f"Unhandled exception in main loop: {str(e)}", exc_info=True)
            
            # Ensure loop runs roughly at screenshot_interval
            elapsed_time = time.time() - start_time
            sleep_duration = max(0, self.screenshot_interval - elapsed_time)
            if self.running.is_set():
                self.running.wait(sleep_duration) # interruptible sleep

    def stop(self):
        logger.info("Stopping AI-Automation loop...")
        self.running.clear()

# --- Example Extension (User can define more like this) ---
class ExampleNotepadExtension(ActionExtension):
    def get_command_name(self) -> str:
        return "NOTEPAD_SAVE_DATE"

    def get_command_description(self) -> str:
        return 'If Notepad is active, types the current date and time, then attempts to save the file. Params: "filename_without_extension"'

    def execute(self, params_str: str, engine: AutomationEngine) -> bool:
        filename_base = params_str.strip().replace('"', '') # Remove quotes if any
        if not filename_base:
            logger.warning(f"{self.get_command_name()}: Filename not provided.")
            return False

        current_app = engine.get_foreground_app()
        if "notepad.exe" not in current_app.lower() and "textedit" not in current_app.lower(): # Basic check
            logger.warning(f"{self.get_command_name()}: Notepad (or similar text editor) not detected as active app ({current_app}). Skipping.")
            return False

        try:
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            pyautogui.typewrite(f"\n\nReport generated on: {now}\n", interval=0.01)
            time.sleep(0.2)
            pyautogui.hotkey('ctrl', 's') # Save
            time.sleep(0.5) # Wait for save dialog
            
            # This part is highly dependent on OS and save dialog specifics
            # For a more robust solution, one might need image recognition here
            # or use OS-specific automation libraries for save dialogs.
            # This is a simplified example.
            pyautogui.typewrite(f"{filename_base}_{now.split(' ')[0]}.txt", interval=0.02)
            time.sleep(0.1)
            pyautogui.press('enter')
            logger.info(f"{self.get_command_name()}: Typed date, attempted save as {filename_base}.txt")
            return True
        except Exception as e:
            logger.error(f"Error during {self.get_command_name()}: {e}")
            return False

# --- Main Execution ---
if __name__ == '__main__':
    logger.info("Application starting...")
    logger.info(f"Using Ollama Vision Model: {OLLAMA_VISION_MODEL}")
    logger.info(f"Using Ollama Embedding Model: {OLLAMA_EMBED_MODEL}")

    # Initialize Ollama client
    try:
        ollama_client = ollama.Client(host=OLLAMA_HOST)
        # Test connection by listing models (optional, but good check)
        ollama_client.list()
        logger.info(f"Successfully connected to Ollama at {OLLAMA_HOST}")
    except Exception as e:
        logger.critical(f"Failed to connect to Ollama at {OLLAMA_HOST}. Ensure Ollama is running and accessible. Error: {e}")
        exit(1)


    # Initialize Databases
    history_db = HistoryDatabase()
    chroma_manager = ChromaDBManager(path=CHROMA_DB_PATH, ollama_host=OLLAMA_HOST, embed_model_name=OLLAMA_EMBED_MODEL)

    # Initialize Obsidian Indexer and run indexing
    if OBSIDIAN_VAULT_PATH:
        obsidian_indexer = ObsidianIndexer(OBSIDIAN_VAULT_PATH, chroma_manager)
        # Consider running indexing in a separate thread or less frequently if it's slow
        threading.Thread(target=obsidian_indexer.index_notes, daemon=True).start()
    else:
        logger.info("OBSIDIAN_VAULT_PATH not set. Skipping Obsidian note indexing.")

    # Initialize Learning System
    learning_system = LearningSystem(history_db, chroma_manager)
    # Periodically run learning analysis (e.g., in a separate thread or less frequently)
    # For this example, run once at startup
    try:
        learning_system.analyze_and_learn_patterns()
    except Exception as e:
        logger.error(f"Error during initial learning system analysis: {e}", exc_info=True)


    # Initialize Automation Engine
    engine = AutomationEngine(history_db, chroma_manager, ollama_client)

    # Register any custom extensions
    example_extension = ExampleNotepadExtension()
    engine.register_extension(example_extension)
    
    # --- Start the main loop ---
    try:
        automation_thread = threading.Thread(target=engine.run_loop, name="AutomationLoop")
        automation_thread.start()
        automation_thread.join() # Wait for the loop to finish (e.g., on stop() or interrupt)
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received in main. Shutting down.")
    except Exception as e:
        logger.critical(f"Fatal error in main execution: {e}", exc_info=True)
    finally:
        logger.info("Cleaning up resources...")
        if 'engine' in locals() and engine.running.is_set():
            engine.stop()
        if 'automation_thread' in locals() and automation_thread.is_alive():
             automation_thread.join(timeout=5) # Wait for thread to stop
        history_db.close()
        # ChromaDB client does not have an explicit close, relies on garbage collection or process end.
        logger.info("Application finished.")