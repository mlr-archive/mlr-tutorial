#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import re
import sys

os.chdir("docs")
with open(os.path.join("..", "mlr-tutorial.md"), "w") as fd:
  fd.write("""
---
documentclass: article
classoption:
  - twoside
title: mlr Tutorial
title-meta: mlr Tutorial
author:
  - Julia Schiffner
  - Bernd Bischl
  - Michel Lang
  - Jakob Richter
  - Zachary M. Jones
  - Philipp Probst
  - Florian Pfisterer
  - Mason Gallo
  - Dominik Kirchhoff
  - Tobias Kühn
  - Lars Kotthoff
---
""")
  with open(os.path.join("..", "mkdocs.yml"), "r") as fd2:
    line = fd2.readline()
    while line:
      if('Appendix' in line):
        break
      elif('.md' in line):
        m = re.search("[^']+\.md", line)
        fname = m.group(0)
        with open(fname, "r") as fd3:
          for lin in fd3:
            if fname != "index.md":
              lin = re.sub(r'^(#+ [A-Za-z0-9])', '#\g<1>', lin)
            fd.write(lin)
          fd.write("\n")
        if('index.md' in line):
          fd.write("\n\n# Basics\n\n")
        if('visualization.md' in line):
          fd.write("\n\n# Advanced\n\n")
        if('hyperpar_tuning_effects.md' in line):
          fd.write("\n\n# Extend\n\n")
      line = fd2.readline()
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

retval = os.system("pandoc --number-sections --latex-engine=xelatex --variable colorlinks=\"true\" --listings -H latex-setup.tex --toc -f markdown+grid_tables+table_captions-implicit_figures -o mlr-tutorial.pdf mlr-tutorial.md")
if retval != 0:
  sys.exit(1)
