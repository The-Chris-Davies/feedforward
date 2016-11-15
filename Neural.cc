#include <iostream>
#include <vector>
#include <cstdlib>
#include <algorithm>
#include <cmath>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/operation.hpp>
#include <boost/numeric/ublas/io.hpp>

using namespace boost::numeric::ublas;

class Net {
	public:
		Net(std::vector <int>);
		matrix<float> feedForward(matrix<float>);
		std::vector<matrix<float> > backProp(matrix<float>, matrix<float>);
		std::vector <int> top;	//the 'topology' of the net.
		void train(unsigned int, std::vector<matrix<float> >, std::vector<matrix<float> >, unsigned int = 1);
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
std::vector<matrix<float> > Net::backProp(matrix<float> inputs, matrix<float> outputs) {
		
	//weighted inputs used for backpropagation - better to call now than to repeatedly call later.
	std::vector<matrix<float> > winps;	//weighted inputs
	std::vector<matrix<float> > acts;	//neuron activations
	acts.push_back(inputs);				//add input neurons - not technically activated, but still required.
	for(unsigned int i = 0; i < weights.size(); ++i) {
		inputs = prod(inputs, weights[i]);
		winps.push_back(inputs);
		inputs = nonlin(inputs);
		acts.push_back(inputs);
	}
	
	//calculate error in output layer
	std::vector<matrix<float> > errors;	//large error store
	
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
		
	//calculate ΔCost for each weight
		//create matrix vector
	std::vector<matrix<float> > dweights;
	for(auto& tempMat : weights){
		dweights.push_back(matrix <float>(tempMat.size1(), tempMat.size2()));
		dweights.back().clear();
	}
		//activated[i]*errors[i]
	for(unsigned int i = 0; i < dweights.size(); ++i){
		dweights[i] = prod(trans(acts[i]), errors[i])/inputs.size1();	//total error is the average error in the batch
	}
	return dweights;
}
void Net::train(unsigned int numLoops, std::vector<matrix<float> > inputs, std::vector<matrix<float> > outputs, unsigned int batchSize) {
	matrix<float> batchInp(batchSize, top[0]), batchOutp(batchSize, top.back());		//containers for batches
	int tempRandInd;		//random index to add to batch
	std::vector<int> randInds;	//random indices for making batches
	std::vector<matrix<float> > dweights;	//weight errors
	
	for(unsigned int numBatches = 0; numBatches < numLoops; ++numBatches){
		//create batches
		randInds.clear();
		batchInp.clear();
		batchOutp.clear();
		for(unsigned int batchFill = 0; batchFill < batchSize; ++batchFill){
			tempRandInd = int(((float)rand() * inputs.size()) / (float)RAND_MAX);
			if(std::find(randInds.begin(), randInds.end(), tempRandInd) != randInds.end())
				--batchFill;
			else{
				randInds.push_back(tempRandInd);
				row(batchInp, batchFill) = row(inputs[tempRandInd], 0);
				row(batchOutp, batchFill) = row(outputs[tempRandInd], 0);
			}
		}
		std::cout << "batch:\t" << batchInp << '\t' << batchOutp << std::endl;
		//run backprop on batches
		dweights = backProp(batchInp, batchOutp);
		
		//train weights!
		for(unsigned int i = 0; i < weights.size(); ++i){
			weights[i] -= dweights[i];
		}
	}
}

int main(){
	Net testNet(std::vector<int>({2, 3, 1}));
	matrix<float> in(4, 2, 1), out(4,1,1);
	std::vector<matrix<float> > inVec(5, in), outVec(5, out);
	
	std::cout << "input:" << std::endl;
	std::cout << in << std::endl;
	std::cout << "weights:" << std::endl;
	for(auto& m: testNet.weights)
		std::cout << m <<std::endl;
	std::cout << "running feed forward" << std::endl;
	std::cout << testNet.feedForward(in) << std::endl;
	std::cout << "training" << std::endl;
	testNet.train(100, inVec, outVec, 2);
	std::cout << "\n\n\n\n\nfeed forward again:" << std::endl;
	std::cout << testNet.feedForward(in) << std::endl;
	return 0;
}
