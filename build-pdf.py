#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import re
import sys

if len(sys.argv) < 4:
  print("Usage: %s [source md file dir] [mkdocs.yml file] [outfile]\n" % sys.argv[0], file = sys.stderr)
  sys.exit(1)

indir = sys.argv[1]
inyml = sys.argv[2]
outfile = sys.argv[3]


with open(outfile, "w") as fd:
  fd.write("""
---
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
  - Janek Thomas
  - Kira Engelhardt
  - Teodora Pandeva
  - Gunnar König
  - Lars Kotthoff
---
""")
  with open(inyml, "r") as fd2:
    line = fd2.readline()
    while line:
      if('Appendix' in line):
        break
      elif('.md' in line):
        m = re.search("[^']+\.md", line)
        fname = m.group(0)
        with open(os.path.join(indir, fname), "r") as fd3:
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

def link_fixer(match):
  file = match.group(1)
  with open(os.path.join(indir, file), 'r') as fd:
    return "(" + re.sub(' ', '-', re.sub(' ', '', fd.readline().rstrip(), 1)).lower() + ")"

with open(outfile, 'r+') as fd:
  data = fd.read()
  out = re.sub(r'\(([a-zA-Z0-9_]+\.md)\)', link_fixer, data)
  fd.seek(0)
  fd.write(out)
  fd.truncate()
