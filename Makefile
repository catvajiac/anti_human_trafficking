EIGENSPOKES= eigenspokes.py data/dense_blocks_3.txt
STREAMING_ALG= streaming_alg.py data/test_streaming_alg.csv
TEST= test.py

all: eigenspokes streaming_alg

eigenspokes: $(EIGENSPOKES)
	@echo "Running example eigenspokes.py..."
	@./eigenspokes.py data/dense_blocks_3.txt

streaming_alg: $(STREAMING_ALG)
	@echo "Running example straming_alg.py..."
	@./streaming_alg.py data/test_streaming_alg.csv

test: $(TEST)
	@./$(TEST)
