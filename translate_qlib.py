import os
import re
import time
import tokenize
import io
import concurrent.futures
from pathlib import Path
from tqdm import tqdm
from deep_translator import GoogleTranslator

REPO_DIR = r"C:\Users\lbw15\Desktop\QLib\qlib"
MAX_WORKERS = 8

class BatchTranslator:
    def __init__(self):
        self.translator = GoogleTranslator(source='auto', target='zh-CN')

    def translate_batch(self, texts):
        if not texts: return []
        
        # Google Translate limit is 5000 chars per request
        # We will split texts into chunks that fit within 4000 chars total
        chunks = []
        current_chunk = []
        current_len = 0
        
        for text in texts:
            if not text.strip() or not any(c.isalpha() for c in text):
                # We will handle these dynamically without translating, but we must keep placeholder
                current_chunk.append(text)
                continue
                
            t_len = len(text)
            if current_len + t_len > 4000 and current_chunk:
                chunks.append(current_chunk)
                current_chunk = [text]
                current_len = t_len
            else:
                current_chunk.append(text)
                current_len += t_len
                
        if current_chunk:
            chunks.append(current_chunk)
            
        results = []
        for chunk in chunks:
            # try to translate batch
            translated = False
            for attempt in range(3):
                try:
                    time.sleep(0.5)
                    # translate_batch can fail if there's a weird string, so fallback to individual if needed
                    try:
                        res = self.translator.translate_batch(chunk)
                        results.extend([r if r else c for r, c in zip(res, chunk)])
                        translated = True
                        break
                    except Exception as batch_e:
                        print(f"Batch failed ({batch_e}), falling back to individual...")
                        # Individual fallback
                        for text in chunk:
                            if any(c.isalpha() for c in text):
                                results.append(self.translator.translate(text))
                            else:
                                results.append(text)
                        translated = True
                        break

                except Exception as e:
                    if '429' in str(e):
                        time.sleep(2 * (attempt + 1))
                    else:
                        break
            if not translated:
                results.extend(chunk) # return original if failed
                
        return results

def process_python_file(file_path):
    translator = BatchTranslator()
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
    except UnicodeDecodeError:
        return
        
    tokens = []
    try:
        generator = tokenize.tokenize(io.BytesIO(code.encode('utf-8')).readline)
        for tok in generator:
            tokens.append(tok)
    except Exception:
        return
        
    translatable_items = []
    texts_to_translate = []
    
    for idx, tok in enumerate(tokens):
        if tok.type == tokenize.COMMENT:
            text = tok.string[1:].strip()
            if any(c.isalpha() for c in text):
                translatable_items.append({'idx': idx, 'type': 'comment', 'prefix': tok.string[:tok.string.find(text)] if text else '# '})
                texts_to_translate.append(text)
        elif tok.type == tokenize.STRING:
            s_val = tok.string
            if s_val.startswith('"""') or s_val.startswith("'''"):
                q_len = 3
                prefix = s_val[:q_len]
                suffix = s_val[-q_len:]
                text = s_val[q_len:-q_len]
                if any(c.isalpha() for c in text) and len(text) > 5:
                     translatable_items.append({'idx': idx, 'type': 'string', 'prefix': prefix, 'suffix': suffix})
                     texts_to_translate.append(text)

    if not texts_to_translate:
        return

    # Batch translate
    translated_texts = translator.translate_batch(texts_to_translate)
            
    # Reconstruct tokens
    result_tokens = list(tokens)
    for i, item in enumerate(translatable_items):
        idx = item['idx']
        tok = result_tokens[idx]
        tr_text = translated_texts[i] if i < len(translated_texts) and translated_texts[i] else texts_to_translate[i]
        
        if item['type'] == 'comment':
            new_string = item['prefix'] + tr_text
        else:
            new_string = item['prefix'] + tr_text + item['suffix']
            
        result_tokens[idx] = tokenize.TokenInfo(tok.type, new_string, tok.start, tok.end, tok.line)
    
    try:
        new_code = tokenize.untokenize(result_tokens).decode('utf-8')
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_code)
    except Exception as e:
        pass

def main():
    py_files = []
    for root, _, files in os.walk(REPO_DIR):
        # Ignore tests or build directories to save time
        if 'tests' in root or '.git' in root or 'docs' in root:
            continue
        for f in files:
            if f.endswith('.py'):
                py_files.append(os.path.join(root, f))
                
    print(f"Translating {len(py_files)} Python files (batched)...")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        list(tqdm(executor.map(process_python_file, py_files), total=len(py_files), desc="PY Files", smoothing=0.1))

    print("Translation completed.")

if __name__ == "__main__":
    main()
