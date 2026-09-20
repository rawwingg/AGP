# Build the AGP MDP / curriculum design notes.
#
#   make          build both PDFs and copy them up into docs/
#   make clean    remove build intermediates (keeps the PDFs)
#   make purge    remove intermediates and PDFs
#
# Requires tectonic (`brew install tectonic`). It fetches any missing LaTeX
# packages on first run, so the first build needs a network connection.

TEX      := tectonic
TEXFLAGS := --keep-intermediates --keep-logs
DOCS     := agp_mdp_formulation agp_staged_ppo_curriculum
PDFS     := $(addsuffix .pdf,$(DOCS))
OUTDIR   := ..
JUNK     := aux log bbl blg out xdv toc

.PHONY: all clean purge
# The committed copies live one level up in docs/, next to nothing else, so
# they are easy to find without digging through the LaTeX sources.
all: $(PDFS)
	cp $(PDFS) $(OUTDIR)/

%.pdf: %.tex agpstyle.sty refs.bib
	$(TEX) $(TEXFLAGS) $<

clean:
	rm -f $(foreach d,$(DOCS),$(foreach e,$(JUNK),$(d).$(e)))

purge: clean
	rm -f $(PDFS) $(foreach p,$(PDFS),$(OUTDIR)/$(p))
