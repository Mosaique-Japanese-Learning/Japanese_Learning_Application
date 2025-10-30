from __future__ import annotations
import json
import os
import re
from pathlib import Path
from typing import Dict, Any, Optional, Callable

import streamlit as st

# ---------- Paths ----------
HERE = Path(__file__).parent
KANJI_JSON = HERE / "merged_kanji.json"
MNEMONICS_JSONL = HERE / "generated_mnemonics_v3.jsonl"
KANJI_SVG_DIR = HERE / "Kanji_SVG"

# ---------- Utilities ----------
_CJK_RE = re.compile(r"[\u3040-\u30FF\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF]+")

def _strip_cjk(s: str) -> str:
    return _CJK_RE.sub("", s or "")

def _read_svg(path: Path) -> Optional[str]:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return None

# Prepare SVG for clean embedding inside Streamlit HTML
_SVG_OPEN_TAG_RE = re.compile(r"<svg[^>]*>", re.IGNORECASE | re.DOTALL)

def _sanitize_svg_for_embed(svg_text: str) -> str:
    """Keep only the <svg>...</svg> fragment and make it responsive.
    Removes XML/DOCTYPE/comments that may render as text when injected.
    Also removes fixed width/height and adds a responsive style attribute.
    """
    if not svg_text:
        return svg_text
    # Extract just the <svg>...</svg>
    start = svg_text.find("<svg")
    end = svg_text.rfind("</svg>")
    frag = svg_text[start:end + len("</svg>")] if (start != -1 and end != -1) else svg_text
    # Strip comments, xml declarations, doctype
    frag = re.sub(r"<!--.*?-->", "", frag, flags=re.DOTALL)
    frag = re.sub(r"<\?xml[^>]*?>", "", frag, flags=re.IGNORECASE)
    frag = re.sub(r"<!DOCTYPE[^>]*>(?:\[[\s\S]*?\])?", "", frag, flags=re.IGNORECASE)
    # Remove fixed size and inject responsive style
    def _strip_size(m: re.Match[str]) -> str:
        tag = m.group(0)
        tag = re.sub(r"\swidth=\"[^\"]*\"", "", tag)
        tag = re.sub(r"\sheight=\"[^\"]*\"", "", tag)
        if "style=" in tag:
            tag = re.sub(
                r"style=\"([^\"]*)\"",
                lambda mm: f"style=\"{mm.group(1)};max-width:420px;width:100%;height:auto;\"",
                tag,
                count=1,
            )
        else:
            tag = tag.replace("<svg", "<svg style=\"max-width:420px;width:100%;height:auto;\"", 1)
        return tag
    frag = _SVG_OPEN_TAG_RE.sub(_strip_size, frag, count=1)
    return frag

def _inject_svg_style(svg_text: str, *, stroke_color: str = "#ffffff", number_color: str = "#ffffff", stroke_width: int = 4, show_numbers: bool = True) -> str:
    """Inject a <style> into the SVG to improve visibility on dark themes.
    We override stroke color/width for paths and fill for stroke numbers.
    """
    style_rules = [
        f"g[id^='kvg:StrokePaths_']{{stroke:{stroke_color} !important;stroke-width:{stroke_width}px !important;}}",
        f"g[id^='kvg:StrokeNumbers_']{{fill:{number_color} !important;font-size:12px !important;}}" if show_numbers else "g[id^='kvg:StrokeNumbers_']{display:none !important;}",
    ]
    style_tag = "<style>" + " ".join(style_rules) + "</style>"
    end_idx = svg_text.rfind("</svg>")
    if end_idx != -1:
        return svg_text[:end_idx] + style_tag + svg_text[end_idx:]
    return style_tag + svg_text

@st.cache_data(show_spinner=False)
def load_kanji_data() -> Dict[str, Any]:
    if not KANJI_JSON.exists():
        st.warning(f"Kanji dataset not found at {KANJI_JSON}")
        return {}
    with KANJI_JSON.open("r", encoding="utf-8") as f:
        data = json.load(f)
    # Expecting a mapping: kanji -> info dict
    return data if isinstance(data, dict) else {}

@st.cache_data(show_spinner=False)
def load_mnemonics_index() -> Dict[str, Dict[str, str]]:
    idx: Dict[str, Dict[str, str]] = {}
    if not MNEMONICS_JSONL.exists():
        return idx
    with MNEMONICS_JSONL.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            k = obj.get("kanji")
            if not k:
                continue
            # last one wins (resume runs)
            idx[k] = {
                "mnemonic": (obj.get("mnemonic") or "").strip(),
                "reminder": (obj.get("reminder") or "").strip(),
            }
    return idx

@st.cache_resource(show_spinner=False)
def load_notebook_generator() -> Optional[Callable[[str], str]]:
    """Load generate_mnemonic from RAG_Mnemonics.ipynb by executing its code cells.
    We avoid executing cells that trigger batch runs. Returns a callable or None if failed.
    """
    nb_path = HERE / "RAG_Mnemonics.ipynb"
    if not nb_path.exists():
        return None
    try:
        with nb_path.open("r", encoding="utf-8") as f:
            nb = json.load(f)
    except Exception:
        return None

    cells = nb.get("cells", [])
    code_parts = []
    for idx, cell in enumerate(cells, start=1):
        # Respect user's request: only use notebook up to Cell 11
        if idx > 11:
            break
        if cell.get("cell_type") != "code":
            continue
        src_list = cell.get("source", [])
        # In some notebooks, source can be a single string
        if isinstance(src_list, str):
            src = src_list
        else:
            src = "".join(src_list)
        # Skip direct execution calls that would run batches or prints
        if "run_batch_with_resume_v3(" in src and "def run_batch_with_resume_v3" not in src:
            continue
        if "print(generate_mnemonic(" in src:
            continue
        code_parts.append(src)

    code = "\n\n".join(code_parts)
    # Execute in an isolated namespace
    ns: Dict[str, Any] = {}
    try:
        exec(code, ns)
        gen = ns.get("generate_mnemonic")
        if callable(gen):
            return gen
        return None
    except Exception:
        return None

# ---------- UI ----------
st.set_page_config(page_title="Kanji Mnemonics", page_icon="🔎", layout="centered")
st.title("Kanji Search with Mnemonics")

with st.sidebar:
    st.header("Data Files")

kanji_data = load_kanji_data()
mnemonic_idx = load_mnemonics_index()

query = st.text_input("Enter a single kanji", max_chars=3, help="Type one kanji character (e.g., 人, 日, 学)")

if query:
    k = query.strip()
    if len(k) != 1:
        st.error("Please enter exactly one kanji character.")
    else:
        info = kanji_data.get(k)
        if not info:
            st.warning("Kanji not found in the dataset.")
        else:
            meanings = info.get("meanings") or []
            # Prefer merged_kanji.json keys, fallback to alternates; also normalize WK-prefixed '!'
            readings_on = (
                info.get("readings_on")
                or info.get("wk_readings_on")
                or info.get("on_readings")
                or info.get("on")
                or []
            )
            readings_kun = (
                info.get("readings_kun")
                or info.get("wk_readings_kun")
                or info.get("kun_readings")
                or info.get("kun")
                or []
            )
            # Normalize readings to list[str]
            if isinstance(readings_on, str):
                readings_on = [readings_on]
            if isinstance(readings_kun, str):
                readings_kun = [readings_kun]
            # Strip leading '!' (WK secondary reading marker)
            readings_on = [str(r).lstrip("!") for r in readings_on]
            readings_kun = [str(r).lstrip("!") for r in readings_kun]

            radicals = info.get("wk_radicals") or []
            stroke_file = info.get("stroke_svg")
            # Compute JLPT (new) display string, replacing Grade with JLPT level
            jlpt_new_val = info.get("jlpt_new")
            jlpt_display: Optional[str] = None
            if jlpt_new_val is not None:
                try:
                    jlpt_display = f"N{int(jlpt_new_val)}"
                except Exception:
                    # Already like "N5" or other string
                    jlpt_display = str(jlpt_new_val).upper()
            else:
                # Fallbacks if jlpt_new missing
                jlpt_fallback = info.get("jlpt") or info.get("jlpt_level") or info.get("jlpt_old")
                if jlpt_fallback is not None:
                    try:
                        jlpt_display = f"N{int(jlpt_fallback)}"
                    except Exception:
                        jlpt_display = str(jlpt_fallback)

            st.markdown(f"## {k}")
            st.caption("Details from main kanji dataset")
            cols = st.columns(2)
            with cols[0]:
                st.markdown("### Meanings")
                if meanings:
                    st.write(", ".join(str(m) for m in meanings))
                else:
                    st.write("—")
                st.markdown("### Radicals")
                st.write(", ".join(str(r) for r in radicals) if radicals else "—")
                if jlpt_display:
                    st.markdown(f"**JLPT:** {jlpt_display}")
            with cols[1]:
                st.markdown("### Readings")
                if readings_on:
                    st.write("Onyomi: " + ", ".join(str(r) for r in readings_on))
                if readings_kun:
                    st.write("Kunyomi: " + ", ".join(str(r) for r in readings_kun))
                if not readings_on and not readings_kun:
                    st.write("—")

            # Stroke order section
            st.markdown("### Stroke order")
            # Controls to improve visibility
            c1, c2, c3 = st.columns([1,1,1])
            with c1:
                hc = st.checkbox("High contrast", value=True, help="Force bright strokes for dark theme")
            with c2:
                show_nums = st.checkbox("Show numbers", value=True)
            with c3:
                sw = st.slider("Stroke width", min_value=2, max_value=8, value=4)

            svg_html = None
            if stroke_file:
                svg_path = KANJI_SVG_DIR / str(stroke_file)
                if svg_path.exists():
                    svg_content = _read_svg(svg_path)
                    if svg_content:
                        clean_svg = _sanitize_svg_for_embed(svg_content)
                        # Apply visual tuning
                        stroke_col = "#ffffff" if hc else "#000000"
                        number_col = "#ffffff" if hc else "#808080"
                        tuned_svg = _inject_svg_style(clean_svg, stroke_color=stroke_col, number_color=number_col, stroke_width=sw, show_numbers=show_nums)
                        # Ensure SVG scales to container width via wrapper too
                        svg_html = f"<div style=\"max-width: 420px;\">{tuned_svg}</div>"
            if svg_html:
                st.markdown(svg_html, unsafe_allow_html=True)
                st.caption("Stroke data: KanjiVG (CC BY-SA 3.0)")
            else:
                st.write("—")

            st.divider()
            st.subheader("Mnemonic (generated via Notebook)")
            line_shown = None
            gen_func = load_notebook_generator()
            if gen_func is None:
                st.warning("Notebook generator unavailable. Ensure RAG_Mnemonics.ipynb is present and dependencies are installed.")
            else:
                with st.spinner("Generating mnemonic using the notebook code (may be slow on CPU)…"):
                    try:
                        line_shown = gen_func(k)
                    except Exception:
                        line_shown = None
                if line_shown:
                    st.success(line_shown)
                else:
                    meanings_str = ", ".join(meanings) if meanings else "meaning"
                    lhs = " + ".join([str(r) for r in radicals]) if radicals else meanings_str
                    fallback = f"{k} = {lhs} → Represents {meanings_str.lower()} through its parts."
                    st.warning("Could not generate via RAG; showing a simple fallback.")
                    st.write(fallback)

            # Optional: show previously saved mnemonic for reference
            rec = mnemonic_idx.get(k)
            if rec and rec.get("mnemonic"):
                with st.expander("Previously saved mnemonic (from JSONL)"):
                    st.write(rec["mnemonic"])
                    if rec.get("reminder"):
                        st.caption(f"Reminder: {rec['reminder']}")

else:
    st.write("Type a kanji above to see its details and mnemonic.")
