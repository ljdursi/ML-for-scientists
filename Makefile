PRES=index

.PHONY: tidy clean

all: $(PRES).html 

BVIMGNAMES=const-bias-variance.png error-vs-degree.png lin-bias-variance.png seventh-bias-variance.png tenth-bias-variance.png twentyth-bias-variance.png
BVIMGS=$(addprefix ./assets/img/bias-variance/,$(BVIMGNAMES))

IMGS=$(BVIMGS)

%.md %.html: %.Rmd $(IMGS)
	Rscript -e "library(slidify); slidify('$<',knit_deck=TRUE,save_payload=TRUE)"

$(BVIMGS): scripts/regression/biasvariance.py
	python $<

tidy:
	rm *~		

clean:
	rm -f $(PRES).html $(PRES).md $(IMGS)
