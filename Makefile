PRES=index

.PHONY: tidy clean

all: $(PRES).html 

BVIMGNAMES=linear-fit.png twentyth-fit.png const-bias-variance.png error-vs-degree.png lin-bias-variance.png seventh-bias-variance.png tenth-bias-variance.png twentyth-bias-variance.png in-sample-error-vs-degree.png
BVIMGS=$(addprefix ./outputs/bias-variance/,$(BVIMGNAMES))
CVIMGS=./outputs/crossvalidation/CV-polynomial.png

BOOTIMGNAMES=area-histogram.png	median-area-histogram.png
BOOTIMGS=$(addprefix ./outputs/bootstrap/,$(BOOTIMGNAMES))

NPIMGNAMES=lowess-fit.png kernel-fit.png kernel-demo.png
NPIMGS=$(addprefix ./outputs/nonparametric/,$(NPIMGNAMES))

DTIMGNAMES=basic.png good-evil.png impurity-plots.png
DTIMGS=$(addprefix ./outputs/classification/,$(DTIMGNAMES))

IMGS=$(BVIMGS) $(CVIMGS) $(BOOTIMGS) $(NPIMGS) $(DTIMGS)

%.md %.html: %.Rmd $(IMGS)
	Rscript -e "library(slidify); slidify('$<',knit_deck=TRUE,save_payload=TRUE)"

$(BVIMGS): scripts/regression/biasvariance.py
	python $<

$(CVIMGS): scripts/regression/crossvalidation.py
	python $<

$(BOOTIMGS): scripts/bootstrap/forestfire.py
	python $<

$(NPIMGS): scripts/regression/lowess.py scripts/regression/kernel.py
	python scripts/regression/lowess.py 
	python scripts/regression/kernel.py 

$(DTIMGS): scripts/classification/decisiontree.py scripts/classification/impurityplot.py
	python scripts/classification/decisiontree.py 
	python scripts/classification/impurityplot.py

tidy:
	rm -f *~
	rm -f scripts/*.pyc scripts/*~ scripts/.*.swp
	rm -f scripts/regression/*.pyc scripts/regression/*~ scripts/regression/.*.swp
	rm -f scripts/bootstrap/*.pyc scripts/bootstrap/*~ scripts/bootstrap/.*.swp
	rm -f scripts/classification/*.pyc scripts/classification/*~ scripts/classification/.*.swp

clean:
	rm -f $(PRES).html $(PRES).md $(IMGS)
