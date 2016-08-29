#!/usr/bin/env python
import os
import re
import sys

os.chdir("docs")
with open(os.path.join("..", "mlr-tutorial.md"), "w") as fd:
  fd.write("""
---
title: mlr Tutorial
---
""")
os.system("for f in `grep -o \"[^']\+\.md\" ../mkdocs.yml`; do (cat \"${f}\"; echo) >> ../mlr-tutorial.md; done")
os.chdir("..")

def link_fixer(match):
  file = match.group(1)
  with open(os.path.join("docs", file), 'r') as fd:
    return "(" + re.sub(' ', '-', re.sub(' ', '', fd.readline().rstrip(), 1)).lower() + ")"

with open('mlr-tutorial.md', 'r+') as fd:
  data = fd.read()
  out = re.sub(r'\(([a-zA-Z0-9_]+\.md)\)', link_fixer, data)
  fd.seek(0)
  fd.write(out)
  fd.truncate()

retval = os.system("pandoc --latex-engine=xelatex --variable colorlinks=\"true\" --listings -H latex-setup.tex --toc -f markdown+grid_tables+table_captions-implicit_figures -o mlr-tutorial.pdf mlr-tutorial.md")
if retval != 0:
  sys.exit(1)
