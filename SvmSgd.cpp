/*
SVM classifier using the Stochastic Gradient Descent. 
This approach followed the one presented in Bottou, Léon. "Large-scale machine learning with stochastic gradient descent." Proceedings of COMPSTAT'2010. Physica-Verlag HD, 2010. 177-186.
To run the code you need OpenCV. For a non-dependent OpenCV implementation you just need to substitute the Mats for float arrays as well as remove all OpenCV dependencies (I pretend to do the same later).

Author: Joao Faro (joaopfaro@gmail.com, joao.faro@isr.uc.pt), Joao Henriques (henriques@isr.uc.pt)
Institute of Systems and Robotics - University of Coimbra
*/

#include "SvmSgd.hpp"

#include <iostream>
#include <fstream>

SvmSgd::SvmSgd(float lambda, float learnRate, uint nIterations){

	// Initialize with random seed
	_randomNumber = 1;

	// Initialize constants
	_slidingWindowSize = 0;
	_nFeatures = 0;
	_predictSlidingWindowSize = 1;

	// Initialize sliderCounter at index 0
	_sliderCounter = 0;

	// Parameters for learning
	_lambda = lambda;  // regularization
	_learnRate = learnRate;  // learning rate (ideally should be large at beginning and decay each iteration)
	_nIterations = nIterations;  // number of training iterations

	// True only in the first predict iteration
	_initPredict = true;

	// Online update flag
	_onlineUpdate = false;
}

SvmSgd::SvmSgd(uint updateFrequency, float lambda, float learnRate, uint nIterations){

	// Initialize with random seed
	_randomNumber = 1;

	// Initialize constants
	_slidingWindowSize = 0;
	_nFeatures = 0;
	_predictSlidingWindowSize = updateFrequency;

	// Initialize sliderCounter at index 0
	_sliderCounter = 0;

	// Parameters for learning
	_lambda = lambda;  // regularization
	_learnRate = learnRate;  // learning rate (ideally should be large at beginning and decay each iteration)
	_nIterations = nIterations;  // number of training iterations

	// True only in the first predict iteration
	_initPredict = true;

	// Online update flag
	_onlineUpdate = true;

	// Learn rate decay: _learnRate = _learnRate * _learnDecay
	_learnRateDecay = 0.1;
}

SvmSgd::~SvmSgd(){

}

SvmSgd* SvmSgd::clone() const{
	return new SvmSgd(*this);
}

void SvmSgd::train(cv::Mat trainFeatures, cv::Mat labels){
	
	// Initialize _nFeatures
	_slidingWindowSize = trainFeatures.rows;
	_nFeatures = trainFeatures.cols;

	float innerProduct;
	// Initialize weights vector with zeros
	if (_weights.size()==0){
		_weights.reserve(_nFeatures);
		for (uint feat = 0; feat < _nFeatures; ++feat){
			_weights.push_back(0.0);
		}
	}

	// Stochastic gradient descent SVM
	for (uint iter = 0; iter < _nIterations; ++iter){
		generateRandomIndex();
		innerProduct = calcInnerProduct(trainFeatures.ptr<float>(_randomIndex));
		int label = (labels.at<int>(_randomIndex,0) > 0) ? 1 : -1; // ensure that labels are -1 or 1
		updateWeights(innerProduct, trainFeatures.ptr<float>(_randomIndex), label );
	}
}

float SvmSgd::predict(cv::Mat newFeature){	
	float innerProduct;

	if (_initPredict){
		_nFeatures = newFeature.cols;
		_slidingWindowSize = _predictSlidingWindowSize;
		_featuresSlider = cv::Mat::zeros(_slidingWindowSize, _nFeatures, CV_32F);
		_initPredict = false;
		_labelSlider = new float[_predictSlidingWindowSize]();
		_learnRate = _learnRate * _learnRateDecay;
	}

	innerProduct = calcInnerProduct(newFeature.ptr<float>(0));
	
	// Resultant label (-1 or 1)
	int label = (innerProduct>=0) ? 1 : -1;

	if (_onlineUpdate){
		// Update the featuresSlider with newFeature and _labelSlider with label
		newFeature.row(0).copyTo(_featuresSlider.row(_sliderCounter));
		_labelSlider[_sliderCounter] = label;
		
		// Update weights with a random index
		if (_sliderCounter == _slidingWindowSize-1){
			generateRandomIndex();
			updateWeights(innerProduct, _featuresSlider.ptr<float>(_randomIndex), _labelSlider[_randomIndex]);
		}

		// _sliderCounter++ if < _slidingWindowSize
		_sliderCounter = (_sliderCounter == _slidingWindowSize-1) ? 0 : (_sliderCounter+1);
	}
		
	return label;
}

void SvmSgd::generateRandomIndex(){
	// Choose random sample, using Mikolov's fast almost-uniform random number
	_randomNumber = _randomNumber * (unsigned long long) 25214903917 + 11;
	_randomIndex = _randomNumber % (unsigned long long) _slidingWindowSize;	
}

float SvmSgd::calcInnerProduct(float *rowDataPointer){
	float innerProduct = 0;
	for (uint feat = 0; feat < _nFeatures; ++feat){
		innerProduct += _weights[feat] * rowDataPointer[feat];
	}
	return innerProduct;
}

void SvmSgd::updateWeights(float innerProduct, float *rowDataPointer, int label){
	if (label * innerProduct > 1) {
		// Not a support vector, only apply weight decay
		for (uint feat = 0; feat < _nFeatures; feat++) {
			_weights[feat] -= _learnRate * _lambda * _weights[feat];
		}
	} else {
		// It's a support vector, add it to the weights
		for (uint feat = 0; feat < _nFeatures; feat++) {
			_weights[feat] -= _learnRate * (_lambda * _weights[feat] - label * rowDataPointer[feat]);
		}
	}
}
