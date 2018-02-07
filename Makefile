
# location of the *.Rmd files
SOURCEDIR = src
# where the local instance of the html, pdf etc. are stored
# NOTE: when changing this, also change mkdocs.yml
LOCALDIR = local
# where TRAVIS is supposed to put the created files
RELEASEDIR = devel

# mkdocs.yml
MKDOCSYML = mkdocs.yml

### intermediate build files:
TEMPDIR = intermediate
# ... for the .md files that get used for PDFs
MDOUT = $(TEMPDIR)/md-files
# ... for the .md files that get used for HTML.
# NOTE: THIS MUST ALSO BE SET IN mkdocs.yml!
MDOUTHTML = $(TEMPDIR)/md-files-for-html
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
# all html input .md files created by convertIt
MDHTMLFILES := $(patsubst $(SOURCEDIR)/%.Rmd,$(MDOUTHTML)/%.md,$(INFILES))

### output files
# html base directory. this overrides mkdocs.yml
# If this changes, the mkdocs.yml should also be updated
# to prevent 'mkdocs serve' from failing.
HTMLBASE = local/html
# html zip
ZIPFILE = $(LOCALDIR)/mlr_tutorial.zip
# pdf file
PDFFILE = $(LOCALDIR)/mlr-tutorial.pdf

MKDIR_P = mkdir -p
PURL_CMD = ./purlIt.Rexec
KNIT_CMD = ./knitIt.Rexec
CONVERT_CMD = ./convertIt.Rexec
BUILDPDF_CMD = ./build-pdf.py
PUBLISH_CMD = ./publish.Rexec

.PHONY: most all pdf html release clean zip

most: pdf html

all: most zip

pdf: $(PDFFILE)

html: $(MDHTMLFILES)
	$(MKDIR_P) $(HTMLBASE)
	mkdocs build -d $(HTMLBASE) -f $(MKDOCSYML)

zip: $(ZIPFILE)


source: $(SOURCEFILES)

release: all
	$(PUBLISH_CMD)
	-rm -r $(RELEASEDIR)/*
	cp -r $(LOCALDIR)/* $(RELEASEDIR)

clean:
	-rm $(PDFMASTERMD)
	-rm $(SRCOUT)/*.R
	-rm $(SRCOUT)/*.tmp
	-rm $(MDOUT)/*.md
	-rm $(MDOUT)/*.tmp
	-rm $(MDOUTHTML)/*.md
	-rm $(MDOUTHTML)/*.tmp
	-rm -r $(HTMLBASE)
	-rm $(ZIPFILE)
	-rm $(PDFFILE)

$(ZIPFILE): html
	cd $(HTMLBASE) ; zip -r "$(shell readlink -f $@)" .

$(PDFFILE): $(PDFMASTERMD)
	pandoc --number-sections --latex-engine=xelatex --variable colorlinks="true" --listings -H latex-setup.tex --toc -f markdown+grid_tables+table_captions-implicit_figures -o "$@" "$<"

$(PDFMASTERMD): $(MDFILES)
	$(MKDIR_P) $(TEMPDIR)
	$(BUILDPDF_CMD) $(MDOUT) $(MKDOCSYML) $(PDFMASTERMD)

# "purl" the .Rmd files into .R files
$(SRCOUT)/%.R: $(SOURCEDIR)/%.Rmd
	$(MKDIR_P) $(SRCOUT)
	$(PURL_CMD) "$<" "$@"

# "knit" the Rmd files into .md files
$(MDOUT)/%.md: $(SOURCEDIR)/%.Rmd
	$(MKDIR_P) $(MDOUT)
	$(KNIT_CMD) "$<" "$@"

# convert md files knit for pdf to md files for html
$(MDOUTHTML)/%.md: $(SOURCEDIR)/%.Rmd $(MDOUT)/%.md $(SRCOUT)/%.R
	$(MKDIR_P) $(MDOUTHTML)
	$(CONVERT_CMD) $? "$@"

