/*
SVM classifier using the Stochastic Gradient Descent. 
This approach followed the one presented in Bottou, Léon. "Large-scale machine learning with stochastic gradient descent." Proceedings of COMPSTAT'2010. Physica-Verlag HD, 2010. 177-186.
To run the code you need OpenCV. For a non-dependent OpenCV implementation you just need to substitute the Mats for float arrays as well as remove all OpenCV dependencies (I pretend to do the same later).

Author: Joao Faro (joaopfaro@gmail.com, joao.faro@isr.uc.pt), Joao Henriques (henriques@isr.uc.pt)
Institute of Systems and Robotics - University of Coimbra
*/

#ifndef SVMSGD_H
#define SVMSGD_H

#include <iostream>
#include <cv.h>

class SvmSgd {

	public:
		SvmSgd(float lambda = 0.000001, float learnRate = 2, uint nIterations = 100000);
		SvmSgd(uint updateFrequency, float lambda = 0.000001, float learnRate = 2, uint nIterations = 100000);
		virtual ~SvmSgd();
		virtual SvmSgd* clone() const;
		virtual void train(cv::Mat trainFeatures, cv::Mat labels);
		virtual float predict(cv::Mat newFeature);
		virtual std::vector<float> getWeights(){ return _weights; };
		virtual void setWeights(std::vector<float> weights){ _weights = weights; };

	private:
		void updateWeights();
		void generateRandomIndex();
		float calcInnerProduct(float *rowDataPointer);
		void updateWeights(float innerProduct, float *rowDataPointer, int label);

		// Vector with SVM weights
		std::vector<float> _weights;

		// Random index generation
		long long int _randomNumber;
		unsigned int _randomIndex;
		
		// Number of features and samples
		unsigned int _nFeatures;
		unsigned int _nTrainSamples;

		// Parameters for learning
		float _lambda;  //regularization
		float _learnRate;  //learning rate
		unsigned int _nIterations; //number of training iterations

		// Vars to control the features slider matrix
		bool _onlineUpdate;
		bool _initPredict;
		uint _slidingWindowSize;
		uint _predictSlidingWindowSize;
		float* _labelSlider;
		float _learnRateDecay;

		// Mat with features slider and correspondent counter
		unsigned int _sliderCounter;
		cv::Mat _featuresSlider;

};

#endif
