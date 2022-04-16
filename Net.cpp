#include <iostream>
#include "Net.h"

Net::Net(vector<vector<double>> images, float learn_scale, vector<vector<int>> required_values)
{
	this->learn_scale = learn_scale;
	this->images = images;
	this->req_val = required_values;
};

void Net::add_layer(int neutron_quantity)
{
	if (this->layers.size() == 0) {
		Layer lay(neutron_quantity, this->images[0].size());
		this->layers.push_back(lay);
	}
	else {
		Layer lay(neutron_quantity, this->layers[this->layers.size() - 1].neurons.size());
		this->layers.push_back(lay);
	}
};

void Net::add_layer(int neutron_quantity, vector<vector<double>> weights)
{
	if (this->layers.size() == 0) {
		Layer lay(neutron_quantity, this->images[0].size(), weights);
		this->layers.push_back(lay);
	}
	else {
		Layer lay(neutron_quantity, this->layers[this->layers.size() - 1].neurons.size(), weights);
		this->layers.push_back(lay);
	}
};

void Net::feed_forward(int image_index)
{
	this->layers[0].feed_forward(this->images, image_index);
	for (size_t i = 1; i < this->layers.size(); i++) {
		this->layers[i].feed_forward(this->layers[i - 1].neurons);
	}
};

void Net::calculate_deltas(int image_index)
{
	this->layers[this->layers.size() - 1].calculate_deltas(this->req_val[image_index]);
	for (int i = this->layers.size() - 2; i >= 0; i--) {
		this->layers[i].calculate_deltas(this->layers[i + 1].neurons);
	}
};

void Net::correct_weights(int image_index)
{
	this->layers[0].correct_weights(this->learn_scale, this->images[image_index]);
	for (size_t i = 1; i < this->layers.size(); i++) {
		this->layers[i].correct_weights(this->learn_scale, this->layers[i - 1].neurons);
	}
};

void Net::learn(int image_index)
{
	this->feed_forward(image_index);
	this->calculate_deltas(image_index);
	this->correct_weights(image_index);
	this->feed_forward(image_index);
};

void Net::get_results()
{
	this->layers[this->layers.size() - 1].get_results();
	cout << '\n';
};

void Net::print() 
{
	for (auto lay : this->layers) {
		lay.print();
	}
	cout << '\n';
};