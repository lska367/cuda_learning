PY   ?= .venv/bin/python
CASE ?= 'gemm/qkv'
SWEEP?= 1,8,64,512,1024
STAMP := $(shell date +%Y%m%d-%H%M%S)

.PHONY: help install build calib bench list sweep attn norm elt csv prof clean

help:
	@echo "make install  安装依赖（uv pip）"
	@echo "make build    编译 CUDA kernel（bench/kernels/csrc → bench/.build）"
	@echo "make calib    标定本机峰值 → bench/results/machine.json"
	@echo "make bench    跑全部 case"
	@echo "make sweep    跑 GEMM 的 M 扫描（roofline 转折点）"
	@echo "make attn|norm|elt   按族跑"
	@echo "make prof     用 nsys 抓一个 case 的 timeline"
	@echo "make list     列出所有 case"

install:
	uv pip install --python $(PY) -r requirements.txt

build:
	$(PY) -c "from bench.kernels import cuda_ops as o; print('cuda ops:', o.available(), o.build_error() or '')"

calib:
	$(PY) -m bench.calib

bench:
	$(PY) -m bench.run

list:
	$(PY) -m bench.run --list

sweep:
	$(PY) -m bench.run -c $(CASE) --sweep T=$(SWEEP)

attn:
	$(PY) -m bench.run -c 'attn/*'

norm:
	$(PY) -m bench.run -c 'norm/*' -c 'elt/*'

elt:
	$(PY) -m bench.run -c 'elt/*'

csv:
	$(PY) -m bench.run --csv bench/results/baseline.csv

prof:
	nsys profile --force-overwrite true -o bench/results/$(STAMP) -t cuda,nvtx \
		$(PY) -m bench.run -c $(CASE) --no-check
	@echo "用 \`nsys stats bench/results/$(STAMP).nsys-rep\` 看汇总"

clean:
	rm -f bench/results/*.nsys-rep bench/results/*.sqlite
