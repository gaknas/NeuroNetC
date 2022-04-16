#include "Layer.h"

Layer::Layer(int neuron_quantity, int weights_count)
{
	for (int i = 0; i < neuron_quantity; i++) {
		this->neurons.push_back(Neuron(weights_count));
	}
}

Layer::Layer(int neuron_quantity, int weights_count, vector<vector<double>> weights)
{
	for (int i = 0; i < neuron_quantity; i++) {
		this->neurons.push_back(Neuron(weights_count, weights[i]));
	}
}

Layer::Layer() {}

void Layer::feed_forward(vector<vector<double>> images, int image_index) 
{
	vector<double> values;
	values = images[image_index];
	for (Neuron& neu : this->neurons) {
		neu.feed_forward(values);
	}
}

void Layer::feed_forward(vector<Neuron> prev_layer)
{
	vector<double> values;
	for (auto neu : prev_layer) {
		values.push_back(neu.value);
	}
	for (Neuron& neu : this->neurons) {
		neu.feed_forward(values);
	}
}

void Layer::calculate_deltas(vector<int> required_values) 
{
	for (size_t i = 0; i < this->neurons.size(); i++) {
		this->neurons[i].calculate_delta(required_values[i]);
	}
}

void Layer::calculate_deltas(vector<Neuron> next_layer) 
{
	for (size_t i = 0; i < this->neurons.size(); i++) {
		this->neurons[i].calculate_delta(i, next_layer);
	}
}

void Layer::correct_weights(float learn_scale, vector<double> image)
{
	for (Neuron &neu : this->neurons) {
		neu.correct_weights(learn_scale, image);
	}
}

void Layer::correct_weights(float learn_scale, vector<Neuron> prev_layer)
{
	for (Neuron& neu : this->neurons) {
		neu.correct_weights(learn_scale, prev_layer);
	}
}

void Layer::get_results() 
{
	for (auto neu : this->neurons) {
		neu.get_result();
	}
}

void Layer::print()
{
	for (auto neu : this->neurons) {
		neu.print();
		cout << '\n';
	}
	cout << '\n';
}
