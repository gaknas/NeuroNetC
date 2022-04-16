#include "Neuron.h"

int Neuron::ids = 1;

void Neuron::get_result() {
	cout << this->value << ' ';
}

void Neuron::print() 
{
	cout << "Нейрон номер " << this->id << '\n';
	cout << "Значениe: " << this->value;
	cout << "\nВеса: ";
	for (auto wei : this->weights) {
		cout << wei << ' ';
	}
	cout << "\nОшибка: ";
	cout << this->delta;
}

void Neuron::take_id() 
{
	this->id = this->ids;
	this->ids++;
}

double Neuron::activation_function(double arg) 
{
	return 1.0 / (1.0 + exp(-arg));
};

Neuron::Neuron(int weights_count) 
{
	this->take_id();
	for (int i = 0; i < weights_count; i++) {
		this->weights.push_back(this->get_random(0, 1));
	}
}

Neuron::Neuron(int weights_count, vector<double> weights)
{
	this->take_id();
	for (int i = 0; i < weights_count; i++) {
		this->weights.push_back(weights[i]);
	}
}

Neuron::Neuron() 
{
	this->take_id();
}

double Neuron::get_random(double start_point, double end_point)
{
	random_device rd;
	mt19937 gen(rd());
	uniform_real_distribution<> distr(start_point, end_point);
	return distr(gen);
}

void Neuron::feed_forward(vector<double> prev_values) 
{
	this->value = this->mult(prev_values, this->weights);
	this->value = this->activation_function(this->value);
}

double Neuron::mult(vector<double> values, vector<double> weights) 
{
	double res = 0;
	for (size_t i = 0; i < values.size(); i++) {
		res += values[i] * weights[i];
	}
	return res;
}

void Neuron::calculate_delta(int req_val) 
{
	this->delta = req_val - this->value;
}

void Neuron::calculate_delta(int index, vector<Neuron> next_layer) 
{
	this->delta = 0;
	for (auto neu : next_layer) {
		this->delta += neu.delta * neu.weights[index];
	}
}

void Neuron::correct_weights(float learn_scale, vector<double> image)
{
	for (size_t i = 0; i < this->weights.size(); i++) {
		this->weights[i] += learn_scale * this->delta * this->value * (1 - this->value) * image[i];
	}
}

void Neuron::correct_weights(float learn_scale, vector<Neuron> prev_layer)
{
	for (size_t i = 0; i < this->weights.size(); i++) {
		this->weights[i] += learn_scale * this->delta * this->value * (1 - this->value) * prev_layer[i].value;
	}
}