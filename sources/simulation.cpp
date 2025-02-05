#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <boost/compute.hpp>

namespace compute = boost::compute;
namespace py = pybind11;
using namespace std;

struct Pendulum
{
	float q1, q2, p1, p2;
};

// Add struct adaptation for Boost.Compute
BOOST_COMPUTE_ADAPT_STRUCT(Pendulum, Pendulum, (q1, q2, p1, p2))


// Function to load OpenCL kernel from file
std::string load_kernel_source(const std::string& file_name) {
	std::ifstream file(file_name);
	if (!file.is_open()) {
		throw std::runtime_error("Failed to open kernel file: " + file_name);
	}
	std::stringstream buffer;
	buffer << file.rdbuf();
	return buffer.str();
}


// Equations of motion for double pendulum
void derivatives(const Pendulum* p, Pendulum* dpdt) {
	const float g = 9.806f;  // gravity

	// Intermediate calculations
	float cos_1 = cos(p->q1 - p->q2);
	float cos_2 = cos_1 * cos_1;
	float sin_1 = sin(p->q1 - p->q2);
	float denom = (2.f - cos_2);
	float denom_2 = denom * denom;
	float prod = p->p1 * p->p2;
	float numer = sin_1 * ( (p->p1 * p->p2 + 2.f * p->p2 * p->p2)*cos_1 - prod * (2 + cos_2)) ;
	float f2 = numer / denom_2;

	// Angular velocity derivatives
	dpdt->q1 = (-       p->p1 + p->p2 * cos_1) / denom;
	dpdt->q2 = (- 2.f * p->p2 + p->p1 * cos_1) / denom;

	// Momentum derivatives
	dpdt->p1 = - f2 + 2 * g * sin(p->q1);
	dpdt->p2 = + f2 +     g * sin(p->q2);
}

py::array_t<float> get_numpy_array(const std::vector<float> &data) {
	// Allocate a NumPy array with the same size
	py::array_t<float> result(data.size());
	auto buf = result.request();
	float* ptr = static_cast<float*>(buf.ptr);

	// Copy data from the vector to the numpy array
	std::memcpy(ptr, data.data(), data.size() * sizeof(float));

	return result;
}

PYBIND11_MODULE(example, m) {
	m.def("get_numpy_array", &get_numpy_array, "Return a numpy array of doubles");
}


py::array_t<float> run_simulation(
		float q1min, float q1max,
		float q2min, float q2max
	) {
	// Simulation parameters
	const unsigned int PIXELS = 1024;
	const float STEP_SIZE = 0.001f;
	const float TOTAL_TIME = 100.0f;
	const unsigned int N = PIXELS * PIXELS;

	// Initialize Boost.Compute context, queue, and device// get the default device
	const compute::device device = compute::system::default_device();
	const compute::context context(device);
	compute::command_queue queue(context, device, compute::command_queue::enable_profiling);

	// Initialize pendulums with Python parameters
	std::vector<Pendulum> pendulums(N);
	for(int i = 0; i < N; ++i) {
		pendulums[i].q1 = q1min + (i / PIXELS + .5f) * (q1max - q1min) / PIXELS;
		pendulums[i].q2 = q2min + (i % PIXELS + .5f) * (q2max - q2min) / PIXELS;
		pendulums[i].p1 = 0.;
		pendulums[i].p2 = 0.;
	}

	// Allocate device buffers
	compute::buffer pendulum_buffer(context, N * sizeof(Pendulum), CL_MEM_READ_ONLY);
	compute::buffer flip_time_buffer(context, N * sizeof(float), CL_MEM_WRITE_ONLY);

	// Copy data to device
	compute::copy(
		pendulums.data(), // Source: host data (Pendulum*)
		pendulums.data() + N, // End of host data
		compute::make_buffer_iterator<Pendulum>(pendulum_buffer, 0), // Destination: buffer iterator
		queue
	);

	// Load and build kernel
	std::string kernel_source = load_kernel_source("../sources/kernel.cl");

	// Compile the kernel program
	compute::program program = compute::program::create_with_source(kernel_source, context);
	try {
		program.build();
	} catch (const compute::opencl_error& e) {
		std::cerr << "OpenCL build error: " << e.what() << std::endl;
		std::cerr << "Build log: " << program.build_log() << std::endl;
		exit(1);
	}

	// Set up kernel
	compute::kernel kernel(program, "flip_time_simulation");
	kernel.set_arg(0, pendulum_buffer);
	kernel.set_arg(1, flip_time_buffer);
	kernel.set_arg(2, sizeof(float), &STEP_SIZE);
	kernel.set_arg(3, sizeof(float), &TOTAL_TIME);
	kernel.set_arg(4, sizeof(int), &N);

	// Execute kernel
	size_t global_work_size = N;
	queue.enqueue_nd_range_kernel(kernel, 1, nullptr, &global_work_size, nullptr);

	// Copy the flip times back to the host
	std::vector<float> flip_times(N);
	compute::copy(
		compute::make_buffer_iterator<float>(flip_time_buffer, 0),
		compute::make_buffer_iterator<float>(flip_time_buffer, N),
		flip_times.begin(),
		queue
	);

	// Return results as numpy array
	py::array_t<float> result(N);
	auto buf = result.request();
	std::memcpy(buf.ptr, flip_times.data(), N * sizeof(float));
	return result;
}

PYBIND11_MODULE(simulation, m) {
	m.def("run_simulation", &run_simulation,
		  "Run double pendulum flip time simulation",
		  py::arg("q1min"), py::arg("q1max"),
		  py::arg("q2min"), py::arg("q2max"));
}