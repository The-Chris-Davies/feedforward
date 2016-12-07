#include <iostream>
#include <vector>
#include <cstdlib>
#include <algorithm>
#include <cmath>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/operation.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <fstream>
#include <cstdint>
#include "./assign.hpp"

using namespace boost::numeric::ublas;

class Net {
	public:
		Net(std::vector <unsigned int>);
		matrix<float> feedForward(matrix<float>);
		std::vector<matrix<float> > backProp(matrix<float>, matrix<float>);
		std::vector <unsigned int> top;	//the 'topology' of the net.
		void train(unsigned int, std::vector<matrix<float> >, std::vector<matrix<float> >, unsigned int = 1);
	//protected:
		std::vector<matrix<float> > weights;
		float nonlin(float, bool = false);
		matrix<float> nonlin(matrix<float>, bool = false);
};
Net::Net(std::vector<unsigned int> topology) {
	std::srand(1);
	top = topology;
	weights.resize(topology.size()-1);
	
	//create weights
	for(unsigned int i = 0; i < topology.size()-1; ++i) {
		weights[i] = matrix<float>(topology[i]+1, topology[i+1]);	//height, width (+2 to add biases)
		for(unsigned int h = 0; h < weights[i].size1(); ++h)
			for(unsigned int w = 0; w < weights[i].size2(); ++w)
				weights[i].insert_element(h, w, ((float)rand() * 2) / (float)RAND_MAX - 1);
	}
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
matrix<float> Net::feedForward(matrix<float> inputs) {
	for(unsigned int i = 0; i < weights.size(); ++i) {
		inputs.resize(inputs.size1(), inputs.size2()+1);
		for(unsigned int h = 0; h < inputs.size1(); ++h)
			inputs.insert_element(h, inputs.size2()-1, 1);
		inputs = prod(inputs, weights[i]);
		inputs = nonlin(inputs);
	}
	return inputs;
}
std::vector<matrix<float> > Net::backProp(matrix<float> inputs, matrix<float> outputs) {
		
	//weighted inputs used for backpropagation - better to call now than to repeatedly call later.
	std::vector<matrix<float> > winps;	//weighted inputs
	std::vector<matrix<float> > acts;	//neuron activations
	acts.push_back(inputs);				//add input neurons - not technically activated, but still required.
	for(unsigned int i = 0; i < weights.size(); ++i) {
		inputs.resize(inputs.size1(), inputs.size2()+1);
		for(unsigned int h = 0; h < inputs.size1(); ++h)
			inputs.insert_element(h, inputs.size2()-1, 1);
		inputs = prod(inputs, weights[i]);
		std::cout << inputs << '\n' << std::endl;
		std::cout << inputs.size1() << '\t' << inputs.size2() << std::endl;
		std::cout << weights[i].size1() << '\t' << weights[i].size2() << std::endl;
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

	std::cout << "errors n' things" << std::endl;
	std::cout << errors[0] << std::endl;
		
	//calculate errors in the rest of the layers
	for(int i = weights.size()-1; i > 0; --i){

	std::cout << "errors in loop" << std::endl;
	std::cout << "errors:\t" << errors[0] << std::endl;
	std::cout << "modded weights:\t" << subrange(weights[i], 0,weights[i].size1()-1, 0,weights[i].size2()) << std::endl;
	std::cout << "nonlin(winps[i-1], true):\t" << nonlin(winps[i-1], true) << std::endl;
		errors.insert(errors.begin(),
			element_prod(
				trans(prod(
					subrange(weights[i], 0,weights[i].size1()-1, 0,weights[i].size2()),
					trans(errors[0]))),
				nonlin(winps[i-1], true)
				));
	}
	std::cout << "done errors! (finally!)" << std::endl;
	
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
		std::cout << numBatches << std::endl;
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
		//run backprop on batches
		dweights = backProp(batchInp, batchOutp);
		
		//train weights!
		for(unsigned int i = 0; i < weights.size(); ++i){
			weights[i] -= dweights[i];
		}
		
	}
}


int reverseInt (int i) {	//for reading ints from file correctly.
    unsigned char c1, c2, c3, c4;

    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;

    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

unsigned int findAns(matrix<float> sol) {
	unsigned int largest;
	float largestSize;
	for(unsigned int h = 0; h < sol.size1(); ++h)
		for(unsigned int w = 0; w < sol.size2(); ++w) {
			if(sol(h, w) > largestSize) {
				largest = w;
				largestSize = sol(h,w);
			}
			
		}
	return largest;
}

int main(){
	//load files
	const unsigned int EXAMPLES(10000);	//number of examples

	std::ifstream images("data/images.idx3-ubyte", std::ios::binary);
	std::ifstream labels("data/labels.idx1-ubyte", std::ios::binary);
	unsigned int labelMagic, imageMagic, labelSize, imageSize, imageRows, imageCols;
	char temp;
	labels.seekg(0, std::ios::beg);
	labels.read((char*)&labelMagic, 4);
	labelMagic = reverseInt(labelMagic);
	labels.read((char*)&labelSize, 4);
	labelSize = reverseInt(labelSize);
	
	images.read((char*)&imageMagic, 4);
	imageMagic = reverseInt(imageMagic);
	images.read((char*)&imageSize, 4);
	imageSize = reverseInt(imageSize);
	
	images.read((char*)&imageRows, 4);
	imageRows = reverseInt(imageRows);
	images.read((char*)&imageCols, 4);
	imageCols = reverseInt(imageCols);
	
	//populate inputs and outputs.
	std::vector<matrix<float> > inputs(EXAMPLES, matrix<float>(1, imageRows*imageCols));
	std::vector<matrix<float> > outputs(EXAMPLES, matrix<float>(1, 10, 0));
	for(unsigned int i = 0; i < EXAMPLES; ++i){
		//output generation
		labels.read(&temp, 1);
		outputs[i].insert_element(0, temp, 1);

		
		//input generation
		for(unsigned int j = 0; j < imageRows*imageCols; ++j){
			images.read(&temp, 1);
			inputs[i].insert_element(0, j, temp);
		}
	}

	Net testNet(std::vector<unsigned int>({imageRows*imageCols, 10, 10}));

	std::cout << "weights:" << std::endl;
	std::cout << "running feed forward" << std::endl;
	std::cout << findAns(outputs[0]) << "\t:\t" << findAns(testNet.feedForward(inputs[0])) << std::endl;
	std::cout << findAns(outputs[1]) << "\t:\t" << findAns(testNet.feedForward(inputs[1])) << std::endl;
	std::cout << findAns(outputs[2]) << "\t:\t" << findAns(testNet.feedForward(inputs[2])) << std::endl;
	std::cout << findAns(outputs[3]) << "\t:\t" << findAns(testNet.feedForward(inputs[3])) << std::endl;
	std::cout << "training" << std::endl;
	testNet.train(100, inputs, outputs, 1);
	std::cout << "feed forward final:" << std::endl;
	std::cout << findAns(outputs[0]) << "\t:\t" << findAns(testNet.feedForward(inputs[0])) << std::endl;
	std::cout << findAns(outputs[1]) << "\t:\t" << findAns(testNet.feedForward(inputs[1])) << std::endl;
	std::cout << findAns(outputs[2]) << "\t:\t" << findAns(testNet.feedForward(inputs[2])) << std::endl;
	std::cout << findAns(outputs[3]) << "\t:\t" << findAns(testNet.feedForward(inputs[3])) << std::endl;
	return 0;
}

/*	print output and input
for(unsigned int j = 0; j < imageRows*imageCols; ++j){
	if(inputs[i](0,j) < 0) std::cout << '@';
	else std::cout << ' ';
	if(j % imageCols == imageCols - 1) std::cout << std::endl;
}
*/
