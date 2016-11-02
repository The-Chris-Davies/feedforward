#include <iostream>
#include <vector>
#include <cstdlib>
#include <cmath>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/operation.hpp>
#include <boost/numeric/ublas/io.hpp>

using namespace boost::numeric::ublas;

class Net {
	public:
		Net(std::vector <int>);
		matrix<float> feedForward(matrix<float>);
		void backProp(matrix<float>, matrix<float>);
		std::vector <int> top;	//the 'topology' of the net.
	//protected:
		std::vector<matrix<float> > weights;
		float nonlin(float, bool = false);
		matrix<float> nonlin(matrix<float>, bool = false);
};
Net::Net(std::vector<int> topology) {
	std::srand(1);
	top = topology;
	weights.resize(topology.size()-1);
	
	for(unsigned int i = 0; i < topology.size()-1; ++i) {
		weights[i] = matrix<float>(topology[i], topology[i+1]);	//height, width
		for(unsigned int h = 0; h < weights[i].size1(); ++h)
			for(unsigned int w = 0; w < weights[i].size2(); ++w)
				weights[i].insert_element(h, w, ((float)rand() * 2) / (float)RAND_MAX - 1);
	}
}
matrix<float> Net::feedForward(matrix<float> inputs) {
	for(unsigned int i = 0; i < weights.size(); ++i) {
		inputs = prod(inputs, weights[i]);
		inputs = nonlin(inputs);
	}
	return inputs;
}
float Net::nonlin(const float inp, bool deriv) {
	float sigmout = (float)(1.0 / (1 + exp((double) -inp)));
	
	if(deriv)
		return sigmout * (1 - sigmout);
	return sigmout;
}
matrix<float> Net::nonlin(matrix<float> inp, bool deriv) {
	//matrix overload for nonlin function
	for(unsigned int h = 0; h < inp.size1(); ++h)
		for(unsigned int w = 0; w < inp.size2(); ++w) {
			inp.insert_element(h, w, nonlin(inp(h, w), deriv));
		}
	return inp;
}
void Net::backProp(matrix<float> inputs, matrix<float> outputs) {
	//weighted inputs used for backpropagation - better to call now than to repeatedly call later.
	std::vector<matrix<float> > winps;	//weighted inputs
	std::vector<matrix<float> > acts;	//neuron activations
	for(unsigned int i = 0; i < weights.size(); ++i) {
		acts.push_back(inputs);
		inputs = prod(inputs, weights[i]);
		winps.push_back(inputs);
		inputs = nonlin(inputs);
	}
	//calculate error in output layer
	std::vector<matrix<float> > errors;	//large error store
	matrix<float> tempErrors;	//errors
	tempErrors.resize(inputs.size1(), top.back(), false);
	
	errors.push_back(		//elementwise product of derivative of cost and derivative of neurons
		element_prod(
			acts.back() - outputs, 
			nonlin(winps.back(), true)));
	
	//calculate errors in the rest of the layers
	for(int i = weights.size()-1; i > 0; --i){
		errors.insert(errors.begin(), 
			element_prod(
				trans(prod(
					weights[i], 
					trans(errors[0]))), 
				nonlin(winps[i-1], true)
				));
	}
	
	//calculate error in ΔCost for each weight
		//create matrix vector
	std::vector<matrix<float> > dweights;
	for(auto& tempMat : weights){
		dweights.push_back(matrix <float>(tempMat.size1(), tempMat.size2()));
		dweights.back().clear();
	}
		//activated[i]*error[i]
	
	
	//print error in weights
	for(auto& m: dweights)
		std::cout << m <<std::endl;
}

int main(){
	Net testNet(std::vector<int>({2, 3, 1}));
	matrix<float> in(3, 2, 1), out(3,1,1);
	
	std::cout << "input:" << std::endl;
	std::cout << in << std::endl;
	std::cout << "weights:" << std::endl;
	for(auto& m: testNet.weights)
		std::cout << m <<std::endl;
	std::cout << "running feed forward" << std::endl;
	std::cout << testNet.feedForward(in) << std::endl;
	std::cout << "running back prop" << std::endl;
	testNet.backProp(in, out);
	return 0;
}
