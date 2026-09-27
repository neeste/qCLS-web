CC = emcc
CFLAGS = -O3 -Wall
EXPORTS = -s EXPORTED_FUNCTIONS="['_generate_stimulus', '_qcls_update_trial', '_qcls_select_bayesian_next', '_qcls_select_isophon_next', '_init_bayesian_state', '_qcls_audiogram_prior', '_qcls_report', '_qcls_pca_fit_report', '_qcls_set_transducer', '_malloc', '_free']"
METHODS = -s EXPORTED_RUNTIME_METHODS="['ccall', 'cwrap', 'HEAPF32']"

all: qcls_core.js

test-pca-fit:
	cc -O3 -std=c99 pca_test/test_pca_fit.c -lm -o /tmp/qcls_test_pca_fit
	/tmp/qcls_test_pca_fit pca_test/ground_truth.csv

qcls_core.js: qcls_core.c index.html
	$(CC) $(CFLAGS) qcls_core.c -o qcls_core.js $(EXPORTS) $(METHODS) -lm

clean:
	rm -f qcls_core.js qcls_core.wasm

em :
	@echo source ~/emsdk/emsdk_env.sh

deploy:
	for f in qcls_core.js qcls_core.wasm index.html guide.html; do \
		curl -T $$f "ftp://audres_deploy%40bonkachen.com:BTNRH1982%21@bonkachen.com/qCLS/$$f"; \
	done
