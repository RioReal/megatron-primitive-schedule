PDFLATEX ?= pdflatex

.PHONY: all clean

all: main.pdf

main.pdf: main.tex figures/fig_cal_normalized_makespan.pdf tables/table_cal_summary.tex tables/table_cal_oracle_summary.tex
	$(PDFLATEX) -interaction=nonstopmode -halt-on-error main.tex
	$(PDFLATEX) -interaction=nonstopmode -halt-on-error main.tex

clean:
	rm -f main.aux main.log main.out main.pdf
