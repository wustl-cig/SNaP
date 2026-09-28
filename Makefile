# Convenience wrappers. Everything here is a one-line python command you can also run
# directly -- see the README.
PY      ?= python
MODEL   ?= celeba_sr
CONFIG  ?= configs/celeba.yaml
N       ?= 100
M       ?= 1

.PHONY: help install data data-celeba data-afhq verify-data weights weights-celeba demo test train eval clean

help:
	@echo "make install        install the python dependencies"
	@echo "make data-celeba    download CelebA into data/celeba      (1.4 GB)"
	@echo "make data-afhq      download AFHQ-Cat into data/afhq_cat  (0.7 GB)"
	@echo "make verify-data    re-check an existing dataset install"
	@echo "make weights-celeba download the 4 CelebA checkpoints     (175 MB)"
	@echo "make weights        download every checkpoint             (2.0 GB)"
	@echo "make demo           run the demo         [MODEL=$(MODEL)] [M=$(M)]"
	@echo "make test           CPU smoke tests"
	@echo "make train          train from scratch   [CONFIG=$(CONFIG)]"
	@echo "make eval           full test-split eval [MODEL=$(MODEL)] [N=$(N)]"

install:
	$(PY) -m pip install -r requirements.txt

data: data-celeba data-afhq
data-celeba:
	$(PY) scripts/download_data.py celeba
data-afhq:
	$(PY) scripts/download_data.py afhq
verify-data:
	$(PY) scripts/download_data.py all --verify-only

weights:
	$(PY) scripts/download_checkpoints.py --all
weights-celeba:
	$(PY) scripts/download_checkpoints.py --group celeba

demo:
	$(PY) demo.py --model $(MODEL) --M $(M)

test:
	$(PY) tests/test_smoke.py

train:
	$(PY) run.py --config $(CONFIG)

eval:
	$(PY) eval_avg.py --config configs/pretrained/$(MODEL).yaml \
		--ckpt checkpoints/$(MODEL).pt --n $(N) --M 1 4 16 100 \
		--out outputs/eval_$(MODEL)

clean:
	rm -rf outputs/ __pycache__ */__pycache__
