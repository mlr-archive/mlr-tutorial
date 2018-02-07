
# location of the *.Rmd files
SOURCEDIR = src
# where the local instance of the html, pdf etc. are stored
LOCALDIR = local
# where TRAVIS is supposed to put the created files
RELEASEDIR = devel

# intermediate build files
TEMPDIR = intermediate
# ... for the .md files that get used for PDFs
MDOUT = $(TEMPDIR)/md-files
# ... for the .md files that get used for HTML
MDOUTHTML = $(TEMPDIR)/md-files-preprocessed
# ... for the purled source files to be included in the html
SRCOUT = $(TEMPDIR)/full_code_src

# all .Rmd files to use
INFILES := $(wildcard $(SOURCEDIR)/*.Rmd)
# all .R files to create by purling
SOURCEFILES := $(patsubst $(SOURCEDIR)/%.Rmd,$(SRCOUT)/%.R,$(INFILES))
# all (pdf input) .md files to create by knit-ing
MDFILES := $(patsubst $(SOURCEDIR)/%.Rmd,$(MDOUT)/%.md,$(INFILES))
# pdf input md file
PDFMASTERMD = $(TEMPDIR)/mlr-tutorial.md

MKDIR_P = mkdir -p
PURL_CMD = ./purlIt.Rexec
KNIT_CMD = ./knitIt.Rexec

.PHONY: all pdf html release clean

all: pdf html source

pdf: $(LOCALDIR)/mlr-tutorial.pdf

html: source

source: $(SOURCEFILES)

release: all $(LOCALDIR)/mlr_tutorial.zip

clean:
	-rm $(PDFMASTERMD)
	-rm $(SRCOUT)/*.R
	-rm $(SRCOUT)/*.tmp
	-rm $(MDOUT)/*.md
	-rm $(MDOUT)/*.tmp

$(LOCALDIR)/mlr_tutorial.zip:

$(LOCALDIR)/mlr-tutorial.pdf: $(PDFMASTERMD)
	pandoc --number-sections --latex-engine=xelatex --variable colorlinks="true" --listings -H latex-setup.tex --toc -f markdown+grid_tables+table_captions-implicit_figures -o "$@" "$<"

$(PDFMASTERMD): $(MDFILES)
	$(MKDIR_P) $(TEMPDIR)
	exit 1

# "purl" the .Rmd files into .R files
$(SRCOUT)/%.R: $(SOURCEDIR)/%.Rmd
	$(MKDIR_P) $(SRCOUT)
	$(PURL_CMD) "$<" "$@"

# "knit" the Rmd files into .md files
$(MDOUT)/%.md: $(SOURCEDIR)/%.Rmd
	$(MKDIR_P) $(MDOUT)
	$(KNIT_CMD) "$<" "$@"

# TODO:
# index: This document --> This webpage
