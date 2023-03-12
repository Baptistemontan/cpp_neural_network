#pragma once

#include <cmath>

template<typename T>
T sigmoid(const T& input) {
    return 1.0 / (1 + std::exp(-1 * input));
}

template<typename T>
T sigmoid_prime(const T& x) {
    return (1 - x) * x;
}

template<typename T>
T relu(const T& x) {
    if(x >= 0.0) {
		return x;
	}
	return 0;
}

template<typename T>
T relu_prime(const T& x) {
    if(x > 0.0) {
		return 1;
	} else {
		return 0;
	}
}