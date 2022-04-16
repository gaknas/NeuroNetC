#pragma once
#include <vector>
#include "Layer.h"
using namespace std;
class Net
{
	vector<Layer> layers;
	vector<vector<int>> req_val;
	vector<vector<double>> images;
public:
	float learn_scale;

	Net(vector<vector<double>> images, float learn_scale, vector<vector<int>> required_values);

	void add_layer(int neuron_quantity);
	void add_layer(int neuron_quantity, vector<vector<double>> weights);

	void feed_forward(int image_index);

	void calculate_deltas(int image_index);

	void correct_weights(int image_index);

	void learn(int image_index);
	
	void get_results();
	void print();
};

