#include "ann.h"
#include <time.h>

int main() {
	system("clear");
	Mat layers(1, 3);
	layers << 2, 3, 1;
	char discard;
	double totaltime = 0;

	clock_t t;
	t = clock();
	std::cout << "[ Testing Theta Initialization ]" << std::endl;
	ann network(layers);
	std::cout << network.theta << std::endl;
	t = clock() - t;
	double time = static_cast<double>(t)/CLOCKS_PER_SEC;
	std::cout << "[ Time Elapsed | " << time << " seconds ]" << std::endl;
	totaltime += time;

	std::cout << "[ Press Enter To Continue ]" << std::endl;
	std::cin.get(discard);
	system("clear");

	Mat vals(5, 1);
	vals << -1, -0.5, 0, 0.5, 1;
	t = clock();
	std::cout << "[ Testing Sigmoid Function ]" << std::endl;
	std::cout << network.sigmoid(vals) << std::endl;
	t = clock() - t;
	time = static_cast<double>(t)/CLOCKS_PER_SEC;
	std::cout << "[ Time Elapsed | " << time << " seconds ]" << std::endl;
	totaltime += time;

	std::cout << "[ Press Enter To Continue ]" << std::endl;
	std::cin.get(discard);
	system("clear");

	t = clock();
	std::cout << "[ Testing Reshape Function ]" << std::endl;
	Mat * T = network.reshape(network.theta);
	for (unsigned i = 0; i < 2; i++) {
		std::cout << T[i] << std::endl;
	}
	t = clock() - t;
	time = static_cast<double>(t)/CLOCKS_PER_SEC;
	std::cout << "[ Time Elapsed | " << time << " seconds ]" << std::endl;
	totaltime += time;

	std::cout << "[ Press Enter To Continue ]" << std::endl;
	std::cin.get(discard);
	system("clear");

	Mat data(10, 2);
	data << 1, 1,
			1, 0,
			0, 1,
			0, 0,
			1, 1,
			1, 0,
			0, 1,
			0, 0,
			1, 1,
			0, 0;

	Mat output(10, 1);
	output << 1,
			0,
			0,
			1,
			1,
			0,
			0,
			1,
			1,
			1;

	t = clock();
	std::cout << "[ Testing Feed Function | 10 Examples | 2 Feature ]" << std::endl;
	Mat pred = network.feed(data);
	std::cout << pred << std::endl;
	t = clock() - t;
	time = static_cast<double>(t)/CLOCKS_PER_SEC;
	std::cout << "[ Time Elapsed | " << time << " seconds ]" << std::endl;
	totaltime += time;

	std::cout << "[ Press Enter To Continue ]" << std::endl;
	std::cin.get(discard);
	system("clear");

	Mat discard2;
	t = clock();
	std::cout << "[ Testing Cost Function | 10 Examples | 2 Features | 1 Output ]" << std::endl;
	double cost = network.costfunction(data, output, 0, discard2);
	std::cout << discard2 << std::endl;
	t = clock() - t;
	time = static_cast<double>(t)/CLOCKS_PER_SEC;
	std::cout << "[ Time Elapsed | " << time << " seconds ]" << std::endl;
	totaltime += time;

	std::cout << "[ Press Enter To Continue ]" << std::endl;
	std::cin.get(discard);
	system("clear");

	t = clock();
	std::cout << "[ Testing Train Function | 10 Examples | 2 Features | 1 Output ]" << std::endl;
	network.train(data, output, 0);
	t = clock() - t;
	time = static_cast<double>(t)/CLOCKS_PER_SEC;
	std::cout << "[ Time Elapsed | " << time << " seconds ]" << std::endl;
	totaltime += time;

	std::cout << "[ Press Enter To Continue ]" << std::endl;
	std::cin.get(discard);
	system("clear");

	t = clock();
	std::cout << "[ Testing Feed Function | 10 Examples | 2 Feature ]" << std::endl;
	pred = network.feed(data);
	std::cout << pow(pred - output, 2).sum() << std::endl;
	t = clock() - t;
	time = static_cast<double>(t)/CLOCKS_PER_SEC;
	std::cout << "[ Time Elapsed | " << time << " seconds ]" << std::endl;
	totaltime += time;

	std::cout << "[ Total Time Elapsed | " << totaltime << " seconds ]" << std::endl;
	return 0;
}
