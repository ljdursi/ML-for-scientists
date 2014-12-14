PRES=index

.PHONY: tidy clean

all: $(PRES).html 

BVIMGNAMES=linear-fit.png twentyth-fit.png const-bias-variance.png error-vs-degree.png lin-bias-variance.png seventh-bias-variance.png tenth-bias-variance.png twentyth-bias-variance.png in-sample-error-vs-degree.png
BVIMGS=$(addprefix ./outputs/bias-variance/,$(BVIMGNAMES))
CVIMGS=./outputs/crossvalidation/CV-polynomial.png

BOOTIMGNAMES=area-histogram.png	median-area-histogram.png
BOOTIMGS=$(addprefix ./outputs/bootstrap/,$(BOOTIMGNAMES))

NPIMGNAMES=lowess-fit.ong
NPIMGS=$(addprefix ./outputs/nonparametric/,$(NPIMGNAMES))

IMGS=$(BVIMGS) $(CVIMGS) $(BOOTIMGS) $(NPIMGS)

%.md %.html: %.Rmd $(IMGS)
	Rscript -e "library(slidify); slidify('$<',knit_deck=TRUE,save_payload=TRUE)"

$(BVIMGS): scripts/regression/biasvariance.py
	python $<

$(CVIMGS): scripts/regression/crossvalidation.py
	python $<

$(BOOTIMGS): scripts/bootstrap/forestfire.py
	python $<

$(NPIMGS): scripts/regression/lowess.py
	python $<

tidy:
	rm -f *~
	rm -f scripts/*.pyc
	rm -f scripts/*~
	rm -f scripts/regression/*.pyc
	rm -f scripts/regression/*~
	rm -f scripts/biasvariance/*.pyc
	rm -f scripts/biasvariance/*~
	rm -f scripts/bootstrap/*.pyc
	rm -f scripts/bootstrap/*~

clean:
	rm -f $(PRES).html $(PRES).md $(IMGS)
