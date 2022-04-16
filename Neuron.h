#pragma once
#include <vector>
#include <iostream>
#include <random>
using namespace std;
class Neuron
{
protected:
	static int ids;
	void take_id();
	double activation_function(double arg);
public:
	int id;

	vector<double> weights;

	double value = 0;
	double delta = 0;

	void get_result();
	void print();

	Neuron(int weights_count);
	Neuron(int weights_count, vector<double> weights);
	Neuron();

	double get_random(double start_point, double end_point);

	void feed_forward(vector<double> values);

	void calculate_delta(int req_val);
	void calculate_delta(int index, vector<Neuron> next_layer);

	void correct_weights(float learn_scale, vector<double> image);
	void correct_weights(float learn_scale, vector<Neuron> prev_layer);

	double mult(vector<double> values, vector<double> weights);
};

