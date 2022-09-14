PHONY: all build clean

build:
	nvcc  -rdc=true -lcudadevrt onehiddenlayerperceptron.cu -o perceptron

clean:
	rm -rf perceptron
