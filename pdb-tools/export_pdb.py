"""
Export the entire GIMP 3 PDB (procedure database) to JSONL for RAG ingestion.

This is NOT a normal script you `python3 export_pdb.py` — it must run inside
a GIMP process, since it needs `gi.repository.Gimp` bound to a live libgimp
instance. Use GIMP's headless python-fu batch interpreter:

    GIMP_PDB_EXPORT_OUT=pdb_export.jsonl \
    gimp-console-3.0 -idf --batch-interpreter python-fu-eval -b - --quit < export_pdb.py

Swap `gimp-console-3.0` for whatever this platform/build calls the console
binary (e.g. `gimp`, `flatpak run org.gimp.GIMP`, `/Applications/GIMP.app/
Contents/MacOS/gimp-console`). `-i` = no GUI, `-d` = no data files (fonts/
brushes/etc, we don't need them), `-f` = no fonts.

Output: one JSON object per line:
  {
    "name": "gimp-image-scale",
    "proc_type": "PLUGIN",              # INTERNAL | PLUGIN | EXTENSION | TEMPORARY
    "blurb": "...",                     # short description
    "help": "...",                      # long description
    "menu_label": "...",
    "authors": "...", "copyright": "...", "date": "...",
    "args": [{"name": ..., "type": ..., "nick": ..., "description": ...}, ...],
    "return_values": [ ... same shape ... ],
    "deprecated": false,                # heuristic: blurb/help mentions deprecation
    "embedding_text": "..."             # precomputed natural-language doc for RAG embedding
  }
"""

import gi
gi.require_version('Gimp', '3.0')
from gi.repository import Gimp
from gi.repository import GObject

import json
import os

OUT_PATH = os.environ.get('GIMP_PDB_EXPORT_OUT', 'gimp_pdb_export.jsonl')


def describe_pspec(pspec):
    """Flatten a GParamSpec into a small dict good for embedding as text."""
    blurb = pspec.get_blurb() or ''

    try:
        # Richer human-readable description: for enums/choices/ranges this
        # includes the valid values, e.g. "{ RUN-NONINTERACTIVE (1) }".
        desc = Gimp.param_spec_get_desc(pspec) or ''
    except Exception:
        desc = ''

    description = ' '.join(p for p in (blurb, desc) if p and p != blurb).strip()
    if not description:
        description = blurb

    return {
        'name': pspec.get_name(),
        'type': GObject.type_name(pspec.value_type),
        'nick': pspec.get_nick() or '',
        'description': description,
    }


# Embedding models truncate past a token budget anyway; cap the source text
# so one pathologically long `help` string can't dominate ingestion time/cost.
MAX_EMBEDDING_TEXT_CHARS = 4000


def build_embedding_text(name, menu_label, blurb, help_text, args):
    """Assemble the natural-language document that gets embedded for RAG.

    Optimized for queries like "change color" or "resize image" to land near
    the right procedure: leads with the human-facing menu label (what a user
    would recognize from GIMP's UI) and the name split into words (so
    "gimp-image-scale" also matches on "image" and "scale"), then blurb/help,
    then argument names+descriptions (so arg-shape questions retrieve too).
    """
    parts = []
    if menu_label:
        parts.append(menu_label.replace('_', '').rstrip('.'))
    parts.append(name.replace('-', ' '))
    if blurb:
        parts.append(blurb)
    if help_text and help_text != blurb:
        parts.append(help_text)

    arg_bits = [f"{a['name']}: {a['description']}" for a in args if a['description']]
    if arg_bits:
        parts.append('Parameters: ' + '; '.join(arg_bits))

    text = '\n'.join(parts).strip()
    return text[:MAX_EMBEDDING_TEXT_CHARS]


def describe_procedure(pdb, name):
    proc = pdb.lookup_procedure(name)
    if proc is None:
        return None

    args = [describe_pspec(p) for p in (proc.get_arguments() or [])]
    return_values = [describe_pspec(p) for p in (proc.get_return_values() or [])]

    blurb = proc.get_blurb() or ''
    help_text = proc.get_help() or ''
    menu_label = proc.get_menu_label() or ''
    deprecated = 'deprecated' in blurb.lower() or 'deprecated' in help_text.lower()

    return {
        'name': name,
        'proc_type': proc.get_proc_type().value_nick.upper(),
        'blurb': blurb,
        'help': help_text,
        'menu_label': menu_label,
        'authors': proc.get_authors() or '',
        'copyright': proc.get_copyright() or '',
        'date': proc.get_date() or '',
        'args': args,
        'return_values': return_values,
        'deprecated': deprecated,
        'embedding_text': build_embedding_text(name, menu_label, blurb, help_text, args),
    }


def main():
    pdb = Gimp.get_pdb()
    names = pdb.query_procedures('.*', '.*', '.*', '.*', '.*', '.*', '.*', '.*')
    names = sorted(names)

    written = 0
    with open(OUT_PATH, 'w') as f:
        for name in names:
            record = describe_procedure(pdb, name)
            if record is None:
                continue
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
            written += 1

    print(f'Wrote {written}/{len(names)} procedures to {OUT_PATH}')


main()
