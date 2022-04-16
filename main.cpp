#include <iostream>
#include <ctime>
#include "Net.h"
using namespace std;

int main() {
	setlocale(LC_ALL, "Russian");
	vector<vector<double>> images = { {0,1,1,0,0,1,1,0,0,0,0,0,0,0,0,0}, {0,0,0,0,0,0,0,0,1,1,0,0,1,1,0,0} };
	vector<vector<int>> required_values = { {1, 0}, {0, 1} };
	Net net(images, 0.01, required_values);
	net.add_layer(3);
	net.add_layer(2);
	net.add_layer(2);
	net.feed_forward(0);
	cout << "Исходное состояние сети:\n";
	for (size_t i = 0; i < images.size(); i++) {
		cout << "Результаты для " << i + 1 << " картинки: ";
		net.feed_forward(i);
		net.calculate_deltas(i);
		net.get_results();
	}
	cout << "Начинаю обучение\n";
	unsigned int start_time = clock();
	for (size_t i = 0; i < 10000; i++) {
		for (size_t j = 0; j < images.size(); j++) {
			net.learn(j);
		}
	}
	net.learn_scale = 0.1;
	for (size_t i = 0; i < 10000; i++) {
		for (size_t j = 0; j < images.size(); j++) {
			net.learn(j);
		}
	}
	cout << "Обучение закончено\n";
	for (size_t i = 0; i < images.size(); i++) {
		cout << "Результаты для " << i + 1 << " картинки: ";
		net.feed_forward(i);
		net.calculate_deltas(i);
		net.get_results();
	}
	unsigned int end_time = clock();
	cout << "Время выполнения: " << end_time - start_time << "мс" << endl;
	system("pause");
	return 0;
}