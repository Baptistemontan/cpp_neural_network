#pragma once

#include "vector.hpp"

template <typename T, std::size_t ROWS, std::size_t COLS>
class Matrix {
public:

	Matrix() = default;
	Matrix(const T& e) {
		for(std::size_t i = 0; i < ROWS; i++) {
			this->data[i] = Vector<T, COLS>(e);
		}
	}

	Matrix(std::initializer_list<std::initializer_list<T>> il) {
		if(il.size() != ROWS) {
            throw std::invalid_argument(string_format("Tried to initialize a matrix with %lu rows but %lu rows where supplied", ROWS, il.size()));
        }
        auto it = il.begin();
        for(std::size_t i = 0; i < ROWS; i++, it++) {
            this->data[i] = Vector<T, COLS>(*it);
        }
	}

	Matrix(std::ifstream& in) {
		std::size_t rows, cols;
		in.read((char*)&rows, sizeof(std::size_t));
		in.read((char*)&cols, sizeof(std::size_t));
		std::cout << "Matrix init with " << rows << " rows and " << cols << " columns" << std::endl;
		if(rows != ROWS || cols != COLS) {
            throw std::invalid_argument(string_format("Tried to initialize a %lux%lu Matrix but binary file contain %lux%lu Matrix", ROWS, COLS, rows, cols));
        }
		in.read((char*)&this->data, sizeof(T) * ROWS * COLS);
	}

	Matrix<T, ROWS, COLS> apply(std::function<T(const T&)>& func) const {
		Matrix<T, ROWS, COLS> out;
		for(std::size_t row = 0; row < ROWS; row++) {
			out[row] = this->data[row].apply(func);
		}
		return out;
	}

	Vector<T, COLS>& operator[](size_t i) {
		if(i >= ROWS) {
			throw std::out_of_range(string_format("Tried to access row %lu but the matrix has %lu rows.", i, ROWS));
		}
		return this->data[i];
	}

	const Vector<T, COLS>& operator[](size_t i) const {
		if(i >= ROWS) {
			throw std::out_of_range(string_format("Tried to access row %lu but the matrix has %lu rows.", i, ROWS));
		}
		return this->data[i];
	}

	Matrix<T, ROWS, COLS> operator+(const Matrix<T, ROWS, COLS> &rhs) const {
		Matrix<T, ROWS, COLS> out;
		for (size_t row = 0; row < ROWS; row++) {
			out[row] = this->data[row] + rhs[row];
		}
		return out;
	}

	Matrix<T, ROWS, COLS>& operator+=(const Matrix<T, ROWS, COLS> &rhs) {
		for (size_t row = 0; row < ROWS; row++) {
			this->data[row] += rhs[row];
		}
		return *this;
	}

	Matrix<T, ROWS, COLS> operator+(const T &rhs) const {
		Matrix<T, ROWS, COLS> out;
		for (size_t row = 0; row < ROWS; row++) {
			out[row] = this->data[row] + rhs;
		}
		return out;
	}

	Matrix<T, ROWS, COLS>& operator+=(const T &rhs) {
		for (size_t row = 0; row < ROWS; row++) {
			this->data[row] += rhs;
		}
		return *this;
	}

	Matrix<T, ROWS, COLS> operator-(const Matrix<T, ROWS, COLS> &rhs) const {
		Matrix<T, ROWS, COLS> out;
		for (size_t row = 0; row < ROWS; row++) {
			out[row] = this->data[row] - rhs[row];
		}
		return out;
	}

	Matrix<T, ROWS, COLS>& operator-=(const Matrix<T, ROWS, COLS> &rhs) {
		for (size_t row = 0; row < ROWS; row++) {
			this->data[row] -= rhs[row];
		}
		return *this;
	}

	Matrix<T, ROWS, COLS> operator-(const T &rhs) const {
		Matrix<T, ROWS, COLS> out;
		for (size_t row = 0; row < ROWS; row++) {
			out[row] = this->data[row] - rhs;
		}
		return out;
	}

	Matrix<T, ROWS, COLS>& operator-=(const T &rhs) {
		for (size_t row = 0; row < ROWS; row++) {
			this->data[row] -= rhs;
		}
		return *this;
	}

	Matrix<T, ROWS, COLS> operator*(const Matrix<T, ROWS, COLS> &rhs) const {
		Matrix<T, ROWS, COLS> out;
		for (size_t row = 0; row < ROWS; row++) {
			out[row] = this->data[row] * rhs[row];
		}
		return out;
	}

	Matrix<T, ROWS, COLS>& operator*=(const Matrix<T, ROWS, COLS> &rhs) {
		for (size_t row = 0; row < ROWS; row++) {
			this->data[row] *= rhs[row];
		}
		return *this;
	}

	Matrix<T, ROWS, COLS> operator*(const T &rhs) const {
		Matrix<T, ROWS, COLS> out;
		for (size_t row = 0; row < ROWS; row++) {
			out[row] = this->data[row] * rhs;
		}
		return out;
	}

	Matrix<T, ROWS, COLS>& operator*=(const T &rhs) {
		for (size_t row = 0; row < ROWS; row++) {
			this->data[row] *= rhs;
		}
		return *this;
	}

	Matrix<T, ROWS, COLS> operator/(const Matrix<T, ROWS, COLS> &rhs) const {
		Matrix<T, ROWS, COLS> out;
		for (size_t row = 0; row < ROWS; row++) {
			out[row] = this->data[row] / rhs[row];
		}
		return out;
	}

	Matrix<T, ROWS, COLS>& operator/=(const Matrix<T, ROWS, COLS> &rhs) {
		for (size_t row = 0; row < ROWS; row++) {
			this->data[row] /= rhs[row];
		}
		return *this;
	}

	Matrix<T, ROWS, COLS> operator/(const T &rhs) const {
		Matrix<T, ROWS, COLS> out;
		for (size_t row = 0; row < ROWS; row++) {
			out[row] = this->data[row] / rhs;
		}
		return out;
	}

	Matrix<T, ROWS, COLS>& operator/=(const T &rhs) {
		for (size_t row = 0; row < ROWS; row++) {
			this->data[row] /= rhs;
		}
		return *this;
	}

	template<size_t RHS_ROWS, size_t RHS_COLS>
	Matrix<T, ROWS, RHS_COLS> dot(const Matrix<T, RHS_ROWS, RHS_COLS>& rhs) const {
		if (COLS != RHS_ROWS) {
			throw std::invalid_argument(string_format("Dot product dimension mismatch, lhs COLS (%lu) != rhs ROWS (%lu)", COLS, RHS_ROWS));
		}
		Matrix<T, ROWS, RHS_COLS> out;
		for (size_t i = 0; i < ROWS; i++) {
			for (size_t j = 0; j < RHS_COLS; j++) {
				T sum = T();
				for (size_t k = 0; k < RHS_ROWS; k++) {
					sum += this->data[i][k] * rhs[k][j];
				}
				out[i][j] = sum;
			}
		}
		return out;
	}

	Vector<T, ROWS> dot(const Vector<T, COLS>& rhs) const {
		Vector<T, ROWS> out;
		for(size_t row = 0; row < ROWS; row++) {
			T sum = T();
			for(size_t col = 0; col < COLS; col++) {
				sum += this->data[row][col] * rhs[col];
			}
			out[row] = sum;
		}
		return out;
	}

	Matrix<T, ROWS * COLS, 1> flatten_vertical() const {
		Matrix<T, ROWS * COLS, 1> out;
		for (size_t row = 0; row < ROWS; row++) {
			for(size_t col = 0; col < COLS; col++) {
				out[row * ROWS + col][0] = this->data[row][col];
			}
		}
		return out;
	}

	Matrix<T, 1, ROWS * COLS> flatten_horizontal() const {
		Matrix<T, 1, ROWS * COLS> out;
		for (size_t row = 0; row < ROWS; row++) {
			for(size_t col = 0; col < COLS; col++) {
				out[0][row * ROWS + col] = this->data[row][col];
			}
		}
		return out;
	}

	Matrix<T, COLS, ROWS> transpose() const {
		Matrix<T, COLS, ROWS> out;
		for (size_t row = 0; row < ROWS; row++) {
			for(size_t col = 0; col < COLS; col++) {
				out[col][row] = this->data[row][col];
			}
		}
		return out;
	}

	std::ofstream& save_binary(std::ofstream& out) const {
		std::size_t rows = ROWS, cols = COLS;
        out.write((const char*)&rows, sizeof(std::size_t));
        out.write((const char*)&cols, sizeof(std::size_t));
		out.write((const char*)&this->data, sizeof(T) * ROWS * COLS);
		return out;
	}

	friend std::ostream &operator<<(std::ostream &output, const Matrix<T, ROWS, COLS> &mat) { 
        output << "[\n";
        for(std::size_t i = 0; i < ROWS - 1; i++) {
            output << "\t" << mat[i] << ",\n";
        }
        if(ROWS != 0) {
            output << "\t" << mat[ROWS - 1];
        }
        output << "\n]";
        return output;
    }

	void randomize(T n) {
		// Pulling from a random distribution of 
		// Min: -1 / sqrt(n)
		// Max: 1 / sqrt(n)
		T min = -1.0 / sqrt(n);
		T max = 1.0 / sqrt(n);
		for (size_t row = 0; row < ROWS; row++) {
			for(size_t col = 0; col < COLS; col++) {
				this->data[row][col] = uniform_distribution(min, max);
			}
		}
	}

private:
	static T uniform_distribution(T low, T high) {
		T difference = high - low; // The difference between the two
		int scale = 10000;
		int scaled_difference = (int)(difference * scale);
		return (T)(low + (1.0 * (rand() % scaled_difference) / scale));
	}

	Vector<T, COLS> data[ROWS];
};