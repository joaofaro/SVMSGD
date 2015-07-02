#include <opencv2/opencv.hpp>
#include "SvmSgd.hpp"

int main (){
    
    // Matrices with the features vectors used in the training phase and the correspondent labels
    cv::Mat trainFeatures; // Matrix with size NxM, with N = number of training examples and M = number of features
    cv::Mat labels; // Matrix with size Nx1

    // Matrix with all features we want to predict (N2xM, with N2 = number of feature vectors to predict)
    cv::Mat predictFeatures;

    // Predicted label for the new feature vector
    int predictedLabel;

    // Initialize object
    SvmSgd SVMSGD;

    // Create simple matrices for demonstration (dummy examples)
    trainFeatures = (cv::Mat_<double>(8,2) << 1, 0, 0, 1, 0, -1, -1, 0, 3, 1, 3, -1, 6, 1, 6, -1 );
    labels = (cv::Mat_<double>(8,1) << -1, -1, -1, -1, 1, 1, 1, 1);
    predictFeatures = (cv::Mat_<double>(6,2) << 4, 5, 1, -2, 10, -1, 6, -1, -2, 1, 9, -2 );

    // Train the Stochastic Gradient Descent SVM
    SVMSGD.train(trainFeatures, labels);

    std::cout << "SVM trained." << std::endl;

    for (int i = 0; i < predictFeatures.rows; ++i){
        // Predict label for the new feature vector (newFeature should be a 1xM matrix)
        predictedLabel = SVMSGD.predict(predictFeatures.row(i));
        std::cout << "Label predicted for feature vector " << i << ": " << predictedLabel << std::endl;
    }

    return 0;
}
