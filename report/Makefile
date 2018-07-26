# Author: Xuedong Shang
SHELL=/bin/sh
PAPER=hpo
BIB=Major

.SUFFIXES: .bib .pdf .tex
.PHONY: clean

run: $(PAPER).pdf

$(PAPER).pdf: $(BIB).bbl $(PAPER).tex
	pdflatex $(PAPER).tex -draftmode
	pdflatex $(PAPER).tex 

$(BIB).bbl: $(PAPER).aux
	bibtex $(PAPER).aux

$(PAPER).aux: $(BIB).bib
	pdflatex $(PAPER).tex -draftmode
	pdflatex $(PAPER).tex -draftmode

clean:
	rm -rf *.aux *.lof *.log *.lot *.toc *.bbl *.blg *.out
