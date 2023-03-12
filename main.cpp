#include <math.h>
#include <time.h>
#include <iostream>
#include "util/img.hpp"
#include "neural/nn.hpp"
#include "neural/activations.hpp"


#define SAVE_FILE_NAME "./testing_net/bin"

int main() {
	srand(time(NULL));

	//TRAINING
	printf("training\n");
	size_t number_training_imgs = 10000; // 10_000
	size_t epochs = 4;
	Img* training_imgs;
	if(csv_to_imgs(&training_imgs, "./data/mnist_test.csv", number_training_imgs)) {
		printf("An error happened while loading the imgs.\n");
		exit(EXIT_FAILURE);
	}
	
	std::function<float(const float&)> activation = [](auto x) {
		return relu(x);
	};
	std::function<float(const float&)> activation_prime = [](auto x) {
		return relu_prime(x);
	};

	NeuralNetwork<float, 784, 300, 10> net;
	net.train_batch(training_imgs, epochs, number_training_imgs, 50, 0.7, 0.9, activation, activation_prime);

	// {
	// 	std::ofstream output_file(SAVE_FILE_NAME, std::ios::out | std::ios::binary | std::ios::trunc);
	// 	if(!output_file) {
	// 		throw std::runtime_error("unable to open the output file");
	// 	}
	// 	net.save_binary(output_file);
	// 	output_file.close();
	// }
	
	imgs_free(training_imgs, number_training_imgs);

	// PREDICTING
	// printf("predicting\n");
	// size_t number_test_imgs = 3000;
	// Img* test_imgs;
	// if(csv_to_imgs(&test_imgs, "data/mnist_test.csv", number_test_imgs)) {
	// 	printf("An error appened while loading the imgs.\n");
	// 	exit(EXIT_FAILURE);
	// }

	// double score = net.predict_imgs(test_imgs, number_test_imgs, activation);
	// printf("Score: %2.3f%%\n", score * 100);
	// imgs_free(test_imgs, number_test_imgs);

	return EXIT_SUCCESS;
}