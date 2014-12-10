PRES=index

.PHONY: .clean

all: $(PRES).html 

%.md %.html: %.Rmd
	Rscript -e "library(slidify); slidify('$<',knit_deck=TRUE,save_payload=TRUE)"

clean:
	rm *~		
