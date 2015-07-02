# Stochastic Gradient Descent SVM

This repository is meant to provide an easy-to-use implementation of the SVM classifier using the Stochastic Gradient Descent. This approach followed the one presented in Bottou, Léon. "Large-scale machine learning with stochastic gradient descent." Proceedings of COMPSTAT'2010. Physica-Verlag HD, 2010. 177-186. To run the code you need OpenCV. For a non-dependent OpenCV implementation you just need to substitute the Mats for float arrays as well as remove all OpenCV dependencies (I pretend to do the same later).

## Code Example

To use the SVM classifier you just need to create the object:
<br />
*SvmSgd SVMSGD;*
<br />
or
<br />
*SvmSgd SVMSGD(updateFrequency);*

And train the classifier with your training features and the corresponding labels:
<br />
*SVMSGD.train(trainFeatures, labels);*

After that you can predict the label of your new feature vector:
<br />
*predictedLabel = SVMSGD.predict(predictFeatures.row(i));*

## Results

The main advantage when compared to others SVM classifiers (like the one present in OpenCV, for example) is the speed performance and the online update of the wheights.
The *out.avi* shows the performance of the presented classifier when used to classify cars using HOG features. As can be seen, the cars are correctly classified. 

## Questions and suggestions

You can contact me by email: joaopfaro@gmail.com

