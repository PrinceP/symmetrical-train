PHONY: all build clean

build:
	nvcc  -std=c++11 -rdc=true -lcudadevrt -Iinclude onehiddenlayerperceptron.cu -o perceptron

clean:
	rm -rf perceptron
