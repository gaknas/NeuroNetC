#pragma once
#include "Neuron.h"
class Layer
{
public:
	vector<Neuron> neurons;

	Layer(int neuron_quantity, int weights_count);
	Layer(int neuron_quantity, int weights_count, vector<vector<double>> weights);
	Layer();

	void feed_forward(vector<vector<double>> images, int image_index);
	void feed_forward(vector<Neuron> prev_layer, int image_index);
	void feed_forward(vector<Neuron> prev_layer);

	void calculate_deltas(vector<int> required_values);
	void calculate_deltas(vector<Neuron> next_layer);

	void correct_weights(float learn_scale, vector<double> image);
	void correct_weights(float learn_scale, vector<Neuron> prev_layer);

	void get_results();
	void print();
};

