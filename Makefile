CC = emcc
CFLAGS = -O3 -Wall
EXPORTS = -s EXPORTED_FUNCTIONS="['_generate_stimulus', '_calculate_bap_next', '_get_loudness_boundaries', '_malloc', '_free']"
METHODS = -s EXPORTED_RUNTIME_METHODS="['ccall', 'cwrap', 'HEAPF32']"

all: qcls_core.js

qcls_core.js: qcls_core.c index.html
	$(CC) $(CFLAGS) qcls_core.c -o qcls_core.js $(EXPORTS) $(METHODS) -lm

clean:
	rm -f qcls_core.js qcls_core.wasm
