import sys
import io
import pandas as pd
import subprocess
import time
from datetime import datetime

# ✅ Force UTF-8 output on Windows (avoids charmap crash)
sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding='utf-8', errors='replace')

# ✅ Safe print to handle any symbol
def safe_print(*args, **kwargs):
    try:
        print(*args, **kwargs)
    except UnicodeEncodeError:
        msg = " ".join(str(a).encode("utf-8", "replace").decode("utf-8") for a in args)
        print(msg, **kwargs)

# ✅ Load radicals
df = pd.read_csv(r"C:\Users\ragur\Japanese_Learning_Application\Mnemonics\japanese-radicals.csv")

# ✅ Generate function using Qwen
def generate_visual_form_ollama(radical, meaning, retries=2):
    prompt = f"""
        You are a visual description assistant for JAPANESE radicals.
        Your task: Describe, in under 10 words, what the radical *looks like* and how it visually connects to its meaning.

        Examples:
        火 (fire) → flames rising upward
        水 (water) → flowing river
        木 (tree) → trunk with spreading branches
        山 (mountain) → three peaks
        日 (sun) → round sun or bright circle

        Now describe visually:
        {radical} ({meaning})
        Output only the short description, no extra text.
    """

    for attempt in range(retries):
        try:
            result = subprocess.run(
                ["ollama", "run", "qwen2.5:1.5b"],
                input=prompt,
                text=True,
                capture_output=True,
                check=True,
                encoding='utf-8'
            )
            response = result.stdout.strip()
            return response
        except subprocess.CalledProcessError as e:
            safe_print(f"⚠️ Ollama error for {radical}: {e.stderr.strip()}")
        except Exception as e:
            safe_print(f"⚠️ Other error for {radical}: {e}")
        time.sleep(1)
    return ""

# ✅ Timestamp for file names
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
csv_path = f"radicals_with_visual_form_{timestamp}.csv"
txt_path = f"radical_visual_descriptions_{timestamp}.txt"

# ✅ Prepare for output
visual_forms = []

# ✅ Text log file
with open(txt_path, "w", encoding="utf-8") as log_file:
    for i, row in df.iterrows():
        radical, meaning = row["Radical"], row["Meaning"]
        safe_print(f"\n🔹 Generating for {radical} ({meaning})...")
        visual = generate_visual_form_ollama(radical, meaning)
        visual_forms.append(visual)

        # ✅ Print and log
        safe_print(f"✨ Visual form: {visual}")
        log_file.write(f"{radical} ({meaning}): {visual}\n")
        time.sleep(1)

# ✅ Save to CSV
df["Visual_Form"] = visual_forms
df.to_csv(csv_path, index=False, encoding='utf-8-sig')

safe_print(f"\n✅ Done — Saved CSV as {csv_path}")
safe_print(f"📝 Log saved as {txt_path}")
