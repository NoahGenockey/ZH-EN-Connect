
"""
LinguaBridge Local - GUI with Document Translation
Desktop application with text and document translation support.
"""

import tkinter as tk
from tkinter import ttk, scrolledtextc, messagebox, filedialog
import threading
import logging
from typing import Optional
from pathlib import Path

try:
    from .inference import TranslationInference, InferenceCache
    from .document_translator import DocumentTranslator, check_dependencies
    from .utils import load_config, setup_logging
except ImportError:
    from inference import TranslationInference, InferenceCache
    from document_translator import DocumentTranslator, check_dependencies
    from utils import load_config, setup_logging


class TranslationApp:
    """
    Desktop GUI with document translation support.
    """
    
    def __init__(self, config: dict):
        """Initialize the application."""
        self.config = config
        self.gui_config = config['deployment']['gui']
        self.logger = logging.getLogger('LinguaBridge.GUI')
        
        # Initialize main window
        self.root = tk.Tk()
        self.root.title("LinguaBridge Local - EN<->ZH Translation")
        self.root.geometry("900x700")
        
        # Initialize engines
        self.inference_engine: Optional[TranslationInference] = None
        self.document_translator: Optional[DocumentTranslator] = None
        self.cache = InferenceCache()
        self.is_translating = False
        
        # Check document support
        self.doc_support = check_dependencies()
        
        # Modernize style
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('.', font=('Segoe UI', 11))
        style.configure('TNotebook', background='#f7f7fa', borderwidth=0)
        style.configure('TNotebook.Tab', font=('Segoe UI', 11, 'bold'), padding=[16, 8], background='#e6e6ed', borderwidth=0)
        style.map('TNotebook.Tab', background=[('selected', '#d0e6fa')])
        style.configure('TFrame', background='#f7f7fa')
        style.configure('TLabel', background='#f7f7fa', font=('Segoe UI', 11))
        style.configure('TLabelframe', background='#f7f7fa', font=('Segoe UI', 11, 'bold'), borderwidth=2, relief='groove')
        style.configure('TLabelframe.Label', font=('Segoe UI', 11, 'bold'))
        style.configure('TButton', font=('Segoe UI', 11, 'bold'), padding=[12, 6], background='#e6e6ed', borderwidth=1, relief='ridge')
        style.map('TButton', background=[('active', '#d0e6fa')])
        style.configure('TProgressbar', background='#4a90e2', troughcolor='#e6e6ed', borderwidth=0)
        style.configure('TRadiobutton', background='#f7f7fa', font=('Segoe UI', 11))
        style.configure('TEntry', font=('Segoe UI', 11))
        style.configure('TScrollbar', background='#e6e6ed')
        self.root.configure(bg='#f7f7fa')
        
        # Setup UI (model loading will be triggered at the end of setup_ui)
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the user interface with tabs."""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Title
        title_label = ttk.Label(
            main_frame,
            text="🌉 LinguaBridge Local - Bidirectional EN<->ZH Translation",
            font=('Arial', 16, 'bold')
        )
        title_label.grid(row=0, column=0, pady=(0, 10), sticky=tk.W)
        
        # Tabbed interface
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Text translation tab
        self.text_tab = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(self.text_tab, text="📝 Text Translation")
        self.setup_text_tab()
        
        # Document translation tab
        self.doc_tab = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(self.doc_tab, text="📚 Document Translation")
        self.setup_document_tab()
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Loading model...")
        status_bar = ttk.Label(
            main_frame,
            textvariable=self.status_var,
            relief=tk.SUNKEN,
            anchor=tk.W
        )
        status_bar.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(10, 0))
        
        # Progress bar
        self.progress_bar = ttk.Progressbar(
            main_frame,
            mode='indeterminate',
            length=200
        )
        self.progress_bar.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=(5, 0))
        self.progress_bar.start(10)

        # Now that all UI elements are created, start model loading
        self.root.after(100, self.load_model_async)

        # Now that all UI elements are created, start model loading
        self.root.after(100, self.load_model_async)
    
    def setup_text_tab(self):
        """Setup text translation tab."""
        self.text_tab.columnconfigure(0, weight=1)
        self.text_tab.rowconfigure(2, weight=1)
        self.text_tab.rowconfigure(4, weight=1)
        
        # Language selection dropdowns and swap button
        direction_frame = ttk.Frame(self.text_tab)
        direction_frame.grid(row=0, column=0, pady=(0, 10), sticky=tk.W)

        ttk.Label(direction_frame, text="Source Language:", font=('Arial', 10, 'bold')).pack(side=tk.LEFT, padx=(0, 5))
        self.source_lang_var = tk.StringVar(value='English')
        self.target_lang_var = tk.StringVar(value='Chinese')
        lang_options = ['English', 'Chinese', 'Japanese']
        self.source_dropdown = ttk.Combobox(direction_frame, textvariable=self.source_lang_var, values=lang_options, state='readonly', width=10)
        self.source_dropdown.pack(side=tk.LEFT, padx=2)

        swap_btn = ttk.Button(direction_frame, text="⇄", width=3, command=self.swap_languages)
        swap_btn.pack(side=tk.LEFT, padx=5)

        ttk.Label(direction_frame, text="Target Language:", font=('Arial', 10, 'bold')).pack(side=tk.LEFT, padx=(5, 5))
        self.target_dropdown = ttk.Combobox(direction_frame, textvariable=self.target_lang_var, values=lang_options, state='readonly', width=10)
        self.target_dropdown.pack(side=tk.LEFT, padx=2)

        self.source_dropdown.bind('<<ComboboxSelected>>', lambda e: self.update_labels())
        self.target_dropdown.bind('<<ComboboxSelected>>', lambda e: self.update_labels())

    def swap_languages(self):
        src = self.source_lang_var.get()
        tgt = self.target_lang_var.get()
        self.source_lang_var.set(tgt)
        self.target_lang_var.set(src)
        self.update_labels()
        
        # Input section
        self.input_label_frame = ttk.LabelFrame(self.text_tab, text="English Input", padding="5")
        self.input_label_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        self.input_label_frame.columnconfigure(0, weight=1)
        self.input_label_frame.rowconfigure(0, weight=1)
        
        self.input_text = scrolledtext.ScrolledText(
            self.input_label_frame,
            wrap=tk.WORD,
            width=70,
            height=10,
            font=('Arial', 11)
        )
        self.input_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Control buttons
        button_frame = ttk.Frame(self.text_tab)
        button_frame.grid(row=3, column=0, pady=10)
        
        self.translate_button = ttk.Button(
            button_frame,
            text="🔄 Translate",
            command=self.translate_text,
            state=tk.DISABLED,
            width=15
        )
        self.translate_button.grid(row=0, column=0, padx=5)
        
        self.clear_button = ttk.Button(
            button_frame,
            text="🗑️ Clear",
            command=self.clear_text,
            width=15
        )
        self.clear_button.grid(row=0, column=1, padx=5)
        
        self.copy_button = ttk.Button(
            button_frame,
            text="📄 Copy Result",
            command=self.copy_translation,
            width=15
        )
        self.copy_button.grid(row=0, column=2, padx=5)
        
        # Output section
        self.output_label_frame = ttk.LabelFrame(self.text_tab, text="Chinese Translation", padding="5")
        self.output_label_frame.grid(row=4, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.output_label_frame.columnconfigure(0, weight=1)
        self.output_label_frame.rowconfigure(0, weight=1)
        
        self.output_text = scrolledtext.ScrolledText(
            self.output_label_frame,
            wrap=tk.WORD,
            width=70,
            height=10,
            font=('Arial', 11),
            state=tk.DISABLED
        )
        self.output_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
    
    def setup_document_tab(self):
        """Setup document translation tab."""
        self.doc_tab.columnconfigure(0, weight=1)
        
        # Direction selector for documents (dropdowns and swap)
        doc_direction_frame = ttk.Frame(self.doc_tab)
        doc_direction_frame.grid(row=0, column=0, pady=(0, 15), sticky=tk.W)

        ttk.Label(doc_direction_frame, text="Source Language:", font=('Arial', 10, 'bold')).pack(side=tk.LEFT, padx=(0, 5))
        self.doc_source_lang_var = tk.StringVar(value='English')
        self.doc_target_lang_var = tk.StringVar(value='Chinese')
        lang_options = ['English', 'Chinese', 'Japanese']
        self.doc_source_dropdown = ttk.Combobox(doc_direction_frame, textvariable=self.doc_source_lang_var, values=lang_options, state='readonly', width=10)
        self.doc_source_dropdown.pack(side=tk.LEFT, padx=2)

        doc_swap_btn = ttk.Button(doc_direction_frame, text="⇄", width=3, command=self.doc_swap_languages)
        doc_swap_btn.pack(side=tk.LEFT, padx=5)

        ttk.Label(doc_direction_frame, text="Target Language:", font=('Arial', 10, 'bold')).pack(side=tk.LEFT, padx=(5, 5))
        self.doc_target_dropdown = ttk.Combobox(doc_direction_frame, textvariable=self.doc_target_lang_var, values=lang_options, state='readonly', width=10)
        self.doc_target_dropdown.pack(side=tk.LEFT, padx=2)

        self.doc_source_dropdown.bind('<<ComboboxSelected>>', lambda e: None)
        self.doc_target_dropdown.bind('<<ComboboxSelected>>', lambda e: None)

    def doc_swap_languages(self):
        src = self.doc_source_lang_var.get()
        tgt = self.doc_target_lang_var.get()
        self.doc_source_lang_var.set(tgt)
        self.doc_target_lang_var.set(src)
        
        # Info frame
        info_frame = ttk.LabelFrame(self.doc_tab, text="📚 Translate Entire Documents", padding="15")
        info_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 20))
        
        info_text = "Upload a PDF or EPUB book/document and get a fully translated version.\n" \
                   "The translation preserves the document structure and formatting."
        ttk.Label(info_frame, text=info_text, wraplength=750).pack()
        
        # File selection
        file_frame = ttk.LabelFrame(self.doc_tab, text="Select Document", padding="15")
        file_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 20))
        file_frame.columnconfigure(1, weight=1)
        
        ttk.Label(file_frame, text="Input File:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        self.file_path_var = tk.StringVar()
        ttk.Entry(file_frame, textvariable=self.file_path_var, width=60).grid(row=0, column=1, sticky=(tk.W, tk.E), padx=5)
        ttk.Button(file_frame, text="Browse...", command=self.browse_file, width=12).grid(row=0, column=2, padx=5)
        
        ttk.Label(file_frame, text="Output File:").grid(row=1, column=0, sticky=tk.W, padx=(0, 10), pady=(10, 0))
        self.output_path_var = tk.StringVar()
        ttk.Entry(file_frame, textvariable=self.output_path_var, width=60).grid(row=1, column=1, sticky=(tk.W, tk.E), padx=5, pady=(10, 0))
        ttk.Button(file_frame, text="Browse...", command=self.browse_output, width=12).grid(row=1, column=2, padx=5, pady=(10, 0))
        
        # Translation button
        self.doc_translate_button = ttk.Button(
            self.doc_tab,
            text="🔄 Translate Document",
            command=self.translate_document,
            state=tk.DISABLED,
            width=25
        )
        self.doc_translate_button.grid(row=2, column=0, pady=20)
        
        # Progress frame
        progress_frame = ttk.LabelFrame(self.doc_tab, text="Translation Progress", padding="15")
        progress_frame.grid(row=3, column=0, sticky=(tk.W, tk.E))
        progress_frame.columnconfigure(0, weight=1)
        
        self.doc_progress_var = tk.StringVar(value="No translation in progress")
        ttk.Label(progress_frame, textvariable=self.doc_progress_var).grid(row=0, column=0, sticky=tk.W)
        
        self.doc_progress_bar = ttk.Progressbar(progress_frame, mode='determinate', length=700)
        self.doc_progress_bar.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(10, 0))
        
        # Supported formats
        support_frame = ttk.LabelFrame(self.doc_tab, text="Supported Formats", padding="15")
        support_frame.grid(row=4, column=0, sticky=(tk.W, tk.E), pady=(20, 0))
        
        pdf_status = "✅ PDF" if self.doc_support['pdf'] else "❌ PDF (not installed)"
        epub_status = "✅ EPUB" if self.doc_support['epub'] else "❌ EPUB (not installed)"
        
        ttk.Label(support_frame, text=f"{pdf_status}  •  {epub_status}").pack()
        
        if not all(self.doc_support.values()):
            install_text = "\nTo enable all formats, run:\npip install -r requirements-documents.txt"
            ttk.Label(support_frame, text=install_text, foreground='red').pack()
    
    def browse_file(self):
        """Browse for input document."""
        filetypes = [("All supported", "*.pdf *.epub")]
        if self.doc_support['pdf']:
            filetypes.append(("PDF files", "*.pdf"))
        if self.doc_support['epub']:
            filetypes.append(("EPUB files", "*.epub"))
        
        filename = filedialog.askopenfilename(
            title="Select document to translate",
            filetypes=filetypes
        )
        
        if filename:
            self.file_path_var.set(filename)
            # Auto-generate output filename
            path = Path(filename)
            output = path.parent / f"{path.stem}_translated{path.suffix}"
            self.output_path_var.set(str(output))
    
    def browse_output(self):
        """Browse for output location."""
        input_file = self.file_path_var.get()
        if not input_file:
            messagebox.showwarning("No Input", "Please select an input file first.")
            return
        
        ext = Path(input_file).suffix
        filename = filedialog.asksaveasfilename(
            title="Save translated document as",
            defaultextension=ext,
            filetypes=[(f"{ext.upper()} files", f"*{ext}")]
        )
        
        if filename:
            self.output_path_var.set(filename)
    
    def translate_document(self):
        """Translate document in background."""
        input_path = self.file_path_var.get()
        output_path = self.output_path_var.get()
        
        if not input_path:
            messagebox.showwarning("No Input", "Please select a document to translate.")
            return
        
        if not output_path:
            messagebox.showwarning("No Output", "Please specify output location.")
            return
        
        if not Path(input_path).exists():
            messagebox.showerror("File Not Found", f"Input file not found:\n{input_path}")
            return
        
        # Get direction from dropdowns
        src = self.doc_source_lang_var.get()
        tgt = self.doc_target_lang_var.get()
        lang_map = {
            ('English', 'Chinese'): 'en-zh',
            ('Chinese', 'English'): 'zh-en',
            ('English', 'Japanese'): 'en-ja',
            ('Japanese', 'English'): 'ja-en',
            ('Chinese', 'Japanese'): 'zh-ja',
            ('Japanese', 'Chinese'): 'ja-zh',
        }
        direction = lang_map.get((src, tgt))
        if not direction:
            messagebox.showerror("Invalid Language Pair", f"Translation from {src} to {tgt} is not supported.")
            return
        
        # Disable button
        self.doc_translate_button.config(state=tk.DISABLED)
        self.doc_progress_bar['value'] = 0
        
        def progress_callback(percent, message):
            self.root.after(0, lambda: self.doc_progress_var.set(message))
            self.root.after(0, lambda: self.doc_progress_bar.configure(value=percent))
        
        def translate():
            try:
                result = self.document_translator.translate_document(
                    input_path,
                    output_path,
                    progress_callback,
                    direction
                )
                self.root.after(0, lambda: self.on_document_complete(result))
            except Exception as e:
                self.logger.error(f"Document translation error: {e}")
                self.root.after(0, lambda: self.on_document_error(str(e)))
        
        thread = threading.Thread(target=translate, daemon=True)
        thread.start()
    
    def on_document_complete(self, output_path):
        """Called when document translation completes."""
        self.doc_translate_button.config(state=tk.NORMAL)
        messagebox.showinfo(
            "Translation Complete",
            f"Document translated successfully!\n\nSaved to:\n{output_path}"
        )
    
    def on_document_error(self, error_msg):
        """Called when document translation fails."""
        self.doc_translate_button.config(state=tk.NORMAL)
        messagebox.showerror("Translation Error", f"Failed to translate document:\n{error_msg}")
    
    # Text translation methods (same as original)
    def load_model_async(self):
        """Load the inference model in background."""
        def load():
            try:
                self.logger.info("Loading inference engine...")
                self.inference_engine = TranslationInference(self.config)
                self.document_translator = DocumentTranslator(self.inference_engine)
                self.root.after(0, self.on_model_loaded)
            except Exception as e:
                self.logger.error(f"Failed to load model: {e}")
                self.root.after(0, lambda e=e: self.on_model_load_error(str(e)))
        
        thread = threading.Thread(target=load, daemon=True)
        thread.start()
    
    def on_model_loaded(self):
        """Called when model is successfully loaded."""
        print("[DEBUG] on_model_loaded called")
        self.progress_bar.stop()
        self.progress_bar.grid_remove()
        self.translate_button.config(state=tk.NORMAL)
        self.doc_translate_button.config(state=tk.NORMAL)
        self.status_var.set("Ready to translate")
        self.logger.info("Model loaded successfully")
    
    def on_model_load_error(self, error_msg):
        """Called when model loading fails."""
        print(f"[DEBUG] on_model_load_error called with error_msg: {error_msg}")
        self.progress_bar.stop()
        self.progress_bar.grid_remove()
        self.status_var.set(f"Error: {error_msg}")
        messagebox.showerror("Model Loading Error", f"Failed to load translation model:\n{error_msg}\n\nCheck your internet connection and model paths in config.yaml.")
    
    def translate_text(self):
        """Translate text using selected source and target languages."""
        if self.is_translating:
            return

        input_text = self.input_text.get("1.0", tk.END).strip()

        if not input_text:
            messagebox.showwarning("Empty Input", "Please enter text to translate.")
            return

        # Get source and target languages
        src = self.source_lang_var.get()
        tgt = self.target_lang_var.get()
        lang_map = {
            ('English', 'Chinese'): 'en-zh',
            ('Chinese', 'English'): 'zh-en',
            ('English', 'Japanese'): 'en-ja',
            ('Japanese', 'English'): 'ja-en',
            ('Chinese', 'Japanese'): 'zh-ja',
            ('Japanese', 'Chinese'): 'ja-zh',
        }
        direction = lang_map.get((src, tgt))
        if not direction:
            messagebox.showerror("Invalid Language Pair", f"Translation from {src} to {tgt} is not supported.")
            return

        # Check cache with direction
        cache_key = f"{direction}:{input_text}"
        cached_result = self.cache.get(cache_key)
        if cached_result:
            self.display_translation(cached_result)
            self.status_var.set("Translation retrieved from cache")
            return

        self.is_translating = True
        self.translate_button.config(state=tk.DISABLED)
        self.status_var.set("Translating...")
        self.progress_bar.grid()
        self.progress_bar.start(10)
        
        def translate():
            try:
                translation = self.inference_engine.translate(input_text, direction)
                self.cache.put(cache_key, translation)
                self.root.after(0, lambda: self.on_translation_complete(translation))
            except Exception as e:
                self.logger.error(f"Translation error: {e}")
                self.root.after(0, lambda: self.on_translation_error(str(e)))
        
        thread = threading.Thread(target=translate, daemon=True)
        thread.start()
    
    def on_translation_complete(self, translation):
        """Called when translation completes."""
        self.progress_bar.stop()
        self.progress_bar.grid_remove()
        self.translate_button.config(state=tk.NORMAL)
        self.is_translating = False
        self.display_translation(translation)
        self.status_var.set("Translation complete")
    
    def on_translation_error(self, error_msg):
        """Called when translation fails."""
        self.progress_bar.stop()
        self.progress_bar.grid_remove()
        self.translate_button.config(state=tk.NORMAL)
        self.is_translating = False
        self.status_var.set(f"Error: {error_msg}")
        messagebox.showerror("Translation Error", f"Failed to translate:\n{error_msg}")
    
    def update_labels(self):
        """Update input/output labels based on selected direction."""
        direction = self.direction_var.get()
        if direction == 'en-zh':
            self.input_label_frame.config(text="English Input")
            self.output_label_frame.config(text="Chinese Translation")
        else:
            self.input_label_frame.config(text="Chinese Input")
            self.output_label_frame.config(text="English Translation")
    
    def display_translation(self, translation):
        """Display translation."""
        self.output_text.config(state=tk.NORMAL)
        self.output_text.delete("1.0", tk.END)
        self.output_text.insert("1.0", translation)
        self.output_text.config(state=tk.DISABLED)
    
    def clear_text(self):
        """Clear all text."""
        self.input_text.delete("1.0", tk.END)
        self.output_text.config(state=tk.NORMAL)
        self.output_text.delete("1.0", tk.END)
        self.output_text.config(state=tk.DISABLED)
        self.status_var.set("Ready to translate")
    
    def copy_translation(self):
        """Copy translation to clipboard."""
        translation = self.output_text.get("1.0", tk.END).strip()
        if translation:
            self.root.clipboard_clear()
            self.root.clipboard_append(translation)
            self.status_var.set("Copied to clipboard")
        else:
            messagebox.showinfo("No Translation", "No translation to copy.")
    
    def run(self):
        """Start the application."""
        self.root.mainloop()


def main():
    """Main entry point."""
    config = load_config()
    logger = setup_logging(config)
    app = TranslationApp(config)
    app.run()


if __name__ == "__main__":
    main()
