#pragma once

#include <functional>
#include <tuple>

#include "../matrix/matrix.hpp"
#include "../util/img.hpp"

template<typename T, std::size_t INPUT_SIZE, std::size_t HIDDEN_SIZE, std::size_t OUTPUT_SIZE>
class NeuralNetwork {
public:
	NeuralNetwork() {
		this->hidden_weights.randomize(HIDDEN_SIZE);
		this->output_weights.randomize(OUTPUT_SIZE);
	}

	NeuralNetwork(std::ifstream& in): hidden_weights(in), output_weights(in) { }

	std::tuple<
		Matrix<T, HIDDEN_SIZE, INPUT_SIZE>,
		Matrix<T, OUTPUT_SIZE, HIDDEN_SIZE>
	> train(
		const Vector<T, INPUT_SIZE>& input,
		const Vector<T, OUTPUT_SIZE>& expected_output,
		std::function<T(const T&)>& activation,
		std::function<T(const T&)>& activation_prime
	) const {
		Vector<T, HIDDEN_SIZE> hidden_output;
		Vector<T, OUTPUT_SIZE> final_output;
		std::tie(hidden_output, final_output) = feed_forward(input, activation);
		Vector<T, HIDDEN_SIZE> hidden_errors;
		Vector<T, OUTPUT_SIZE> output_errors;
		std::tie(hidden_errors, output_errors) = find_errors(expected_output, final_output);
		return back_propagate(hidden_errors, output_errors, hidden_output, final_output, input, activation_prime);
	}

	std::tuple<
		Matrix<T, HIDDEN_SIZE, INPUT_SIZE>,
		Matrix<T, OUTPUT_SIZE, HIDDEN_SIZE>
	> train_mini_batch(
		Img* imgs,
		std::function<T(const T&)>& activation,
		std::function<T(const T&)>& activation_prime,
		std::size_t mini_batch_size
	) const {
		Matrix<T, HIDDEN_SIZE, INPUT_SIZE> hidden_delta_avg(0);
		Matrix<T, OUTPUT_SIZE, HIDDEN_SIZE> output_delta_avg(0);
		for(size_t i = 0; i < mini_batch_size; i++) {
			Img* cur_img = imgs + i;
			Vector<T, OUTPUT_SIZE> expected_output(0);
			expected_output[cur_img->label] = 1;

			Matrix<T, HIDDEN_SIZE, INPUT_SIZE> hidden_delta;
			Matrix<T, OUTPUT_SIZE, HIDDEN_SIZE> output_delta;

			std::tie(hidden_delta, output_delta) = train(cur_img->img_data, expected_output, activation, activation_prime);

			hidden_delta_avg += hidden_delta;
			output_delta_avg += output_delta;
		}

		return std::make_tuple(hidden_delta_avg, output_delta_avg);
	}

	void train_batch_inner(
		Img* imgs, 
		const T& lr,
		std::function<T(const T&)>& activation,
		std::function<T(const T&)>& activation_prime,
		std::size_t mini_batch_size
	) {
		Matrix<T, HIDDEN_SIZE, INPUT_SIZE> hidden_delta_avg;
		Matrix<T, OUTPUT_SIZE, HIDDEN_SIZE> output_delta_avg;

		std::tie(hidden_delta_avg, output_delta_avg) = train_mini_batch(imgs, activation, activation_prime, mini_batch_size);

		this->hidden_weights += hidden_delta_avg * (lr / mini_batch_size);
		this->output_weights += output_delta_avg * (lr / mini_batch_size);
	}

	void train_batch(
		Img* imgs,
		std::size_t epochs,
		std::size_t batch_size,
		std::size_t mini_batch_size,
		T lr,
		const T& lr_coef,
		std::function<T(const T&)>& activation,
		std::function<T(const T&)>& activation_prime
	) {
		for(std::size_t e = 1; e <= epochs; e++) {
			for (std::size_t i = 0; i < batch_size; i += mini_batch_size) {
				std::cout << "Epoch " << e << '/' << epochs << ", Img Batch No. " << (i / mini_batch_size) + 1 << '/' << batch_size / mini_batch_size << std::endl;
				train_batch_inner(imgs + i, lr, activation, activation_prime, mini_batch_size);
			}
			lr *= lr_coef;
		}
	}

	Vector<T, OUTPUT_SIZE> predict(const Vector<T, INPUT_SIZE>& input, std::function<T(const T&)>& activation) const {
		auto feed = feed_forward(input, activation);
		auto res = std::get<1>(feed);
		return res.softmax();
	}

	std::size_t predict_img(const Img& img, std::function<T(const T&)>& activation) const {
		Vector<T, OUTPUT_SIZE> res = predict(img.img_data, activation);
		return res.argmax();
	}

	double predict_imgs(Img* imgs, std::size_t n_imgs, std::function<T(const T&)>& activation) const {
		std::size_t n_correct = 0;
		for(std::size_t i = 0; i < n_imgs; i++) {
			Img& img = imgs[i];
			std::size_t prediction = predict_img(img, activation);
			if(prediction == img.label) {
				n_correct++;
			}
		}
		return 1.0 * n_correct / n_imgs;
	}

	std::ofstream& save_binary(std::ofstream& out) const {
		this->hidden_weights.save_binary(out);
		this->output_weights.save_binary(out);
		return out;
	}


private:

	std::tuple<Vector<T, HIDDEN_SIZE>, Vector<T, OUTPUT_SIZE>> feed_forward(
		const Vector<T, INPUT_SIZE>& input, 
		std::function<T(const T&)>& activation
	) const {
		Vector<T, HIDDEN_SIZE> hidden_output_unactivated = this->hidden_weights.dot(input);
		Vector<T, HIDDEN_SIZE> hidden_output = hidden_output_unactivated.apply(activation);
		Vector<T, OUTPUT_SIZE> final_output_unactivated = this->output_weights.dot(hidden_output);
		Vector<T, OUTPUT_SIZE> final_output = final_output_unactivated.apply(activation);
		return std::make_tuple(hidden_output, final_output);
	}

	std::tuple<Vector<T, HIDDEN_SIZE>, Vector<T, OUTPUT_SIZE>> find_errors(
		const Vector<T, OUTPUT_SIZE>& expected_output, 
		const Vector<T, OUTPUT_SIZE>& final_output
	) const {
		Matrix<T, HIDDEN_SIZE, OUTPUT_SIZE> transposed_mat = this->output_weights.transpose();
		Vector<T, OUTPUT_SIZE> output_errors = expected_output - final_output;
		Vector<T, HIDDEN_SIZE> hidden_errors = transposed_mat.dot(output_errors);
		return std::make_tuple(hidden_errors, output_errors);
	}

	template<std::size_t WEIGHTS_ROWS, std::size_t WEIGHTS_COLS> 
	static Matrix<T, WEIGHTS_ROWS, WEIGHTS_COLS> back_propagate_core(
		const Vector<T, WEIGHTS_ROWS>& output, 
		const Vector<T, WEIGHTS_ROWS>& errors,
		const Vector<T, WEIGHTS_COLS>& input,
		std::function<T(const T&)>& activation_prime
	) {
		Vector<T, WEIGHTS_ROWS> primed_output = output.apply(activation_prime);
		Vector<T, WEIGHTS_ROWS> multiplied_output = errors * primed_output;
		return multiplied_output.dot(input);
	}

	static std::tuple<
		Matrix<T, HIDDEN_SIZE, INPUT_SIZE>,
		Matrix<T, OUTPUT_SIZE, HIDDEN_SIZE>
	> back_propagate(
		const Vector<T, HIDDEN_SIZE>& hidden_errors, 
		const Vector<T, OUTPUT_SIZE>& output_errors,
		const Vector<T, HIDDEN_SIZE>& hidden_output, 
		const Vector<T, OUTPUT_SIZE>& final_output,
		const Vector<T, INPUT_SIZE>& input,
		std::function<T(const T&)>& activation_prime
	) {
		Matrix<T, OUTPUT_SIZE, HIDDEN_SIZE> output_delta = back_propagate_core(
			final_output, 
			output_errors, 
			hidden_output, 
			activation_prime
		);
		Matrix<T, HIDDEN_SIZE, INPUT_SIZE> hidden_delta = back_propagate_core(
			hidden_output, 
			hidden_errors, 
			input, 
			activation_prime
		);

		return std::make_tuple(hidden_delta, output_delta);
	}

	Matrix<T, HIDDEN_SIZE, INPUT_SIZE> hidden_weights;
	Matrix<T, OUTPUT_SIZE, HIDDEN_SIZE> output_weights;
};