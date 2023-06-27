#include <iostream>
#include <fstream>
#include <thread>
#include <iomanip>
#include <chrono>
#include <ratio>
#include <functional>
#include <random>
#include <CL/sycl.hpp>
using namespace cl::sycl;

void random_init(buffer<float, 2>& buf) {
	host_accessor m{ buf ,read_write };
	static std::default_random_engine generator(0);
	static std::uniform_real_distribution<float> distribution(-1.0, 1.0);
	int n = m.get_range()[0];
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < i; j++)
			m[i][j] = 0;
		m[i][i] = 1.0;
		for (int j = i + 1; j < n; j++)
			m[i][j] = distribution(generator);
	}
	for (int k = 0; k < n; k++) {
		for (int i = k + 1; i < n; i++) {
			for (int j = 0; j < n; j++) {
				m[i][j] += m[k][j];
			}
		}
	}
}
void serial(buffer<float, 2>& buf, queue& q) {
	host_accessor m{ buf ,read_write };
	int n = m.get_range()[0];
	for (int k = 0; k < n; k++) {
		for (int j = k + 1; j < n; j++) {
			m[k][j] = m[k][j] / m[k][k];
		}
		m[k][k] = 1;
		for (int i = k + 1; i < n; i++) {
			for (int j = k + 1; j < n; j++) {
				m[i][j] = m[i][j] - m[i][k] * m[k][j];
			}
			m[i][k] = 0;
		}
	}
}
void gauss_gpu(buffer<float, 2>& N, queue& q) {
    int n = N.get_range()[0];
    for (int k = 0; k < n; k++) {
        // 除法
        q.submit([&](handler& h) {
            accessor m{ N, h, read_write };
            h.parallel_for(range(n - k), [=](id<1> idx) {
                int j = k + idx[0];
                m[k][j] = m[k][j] / m[k][k];
            });
        });

        // 消去
        q.submit([&](handler& h) {
            accessor m{ N, h, read_write };
            h.parallel_for(range(n - (k + 1), n - (k + 1)), [=](id<2> idx) {
                int i = k + 1 + idx[0];
                int j = k + 1 + idx[1];
                m[i][j] = m[i][j] - m[i][k] * m[k][j];
            });
        });

        // 置零
        q.submit([&](handler& h) {
            accessor m{ N, h, write };
            h.parallel_for(range(n - (k + 1)), [=](id<1> idx) {
                int i = k + 1 + idx[0];
                m[i][k] = 0;
            });
        });
    }
    q.wait_and_throw();
}
using gauss_func = std::function<void(buffer<float, 2>&, queue&)>;
double runaverage(int n, const gauss_func& func,  queue& q) {
	buffer<float, 2> buf(range(n, n));
	random_init(buf);
	func(buf, q);
	std::chrono::duration<double, std::milli> elapsed{};
	for (int i = 0; i < 10; i++) {
		random_init(buf);
		auto start = std::chrono::high_resolution_clock::now();
		func(buf,q);
		auto end = std::chrono::high_resolution_clock::now();
		elapsed += end - start;
	}
	return elapsed.count() / 10;
}

void runrun(const std::vector<gauss_func>& gauss_funcs,const std::vector<std::string>& names,int iter0,int iter1,queue& q) {

	std::cout << "问题规模N,";
	for (auto& name : names) {
		std::cout << name << ",";
	}
	std::cout << std::endl;
	for (int n = iter0; n <= iter1; n *= 2) {
		std::cout << n << ",";
		for (auto& func : gauss_funcs) {
			std::cout << runaverage(n, func, q) << ",";
		}
		std::cout << std::endl;
	}
}

int main() {

	queue q;
	device my_device = q.get_device();
	std::cout << "Device: " << my_device.get_info<info::device::name>() << std::endl;
	std::vector<gauss_func> gauss_funcs = {serial,gauss_gpu,};
	std::vector<std::string> names = {"serial","gpu",};
	runrun(gauss_funcs, names, 4, 4096, q);
}
