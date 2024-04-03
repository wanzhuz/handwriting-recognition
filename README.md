# USPS Handwriting Recognition

## Abstract
In today’s digitalized world, image and pattern recognition play a vital role. This significance stems from their ability to automate
mundane tasks that, although manageable by humans, can be viewed as inefficient uses of their time, such as digit or facial recognition.
The objective of this project is to compare the effectiveness and accuracy of two algorithms: a basic algorithm and an SVD (Singular
Value Decomposition) basis algorithm. Specifically, we aim to automate the identification of handwritten digits using data from the
United States Postal Service (USPS). Automating this task is important because while humans are capable of recognizing digits,
it is a mundane task that becomes prone to errors over time. In contrast, computers can perform this task much faster and with
predetermined accuracy. By finding an image recognition algorithm that performs well, we can accomplish more in less time. This
report explores the effectiveness of these two algorithms in achieving this objective, and we later conclude that SVD performs
significantly better than our simple classification algorithm.

## Data Set
We have access to a database named ”USPS.mat” comprising handwritten digit data. This database consists of two main sets:
  * training set
  * test set

Each set contains patterns of digits along with their corresponding labels. The training set consists of two components:
1. Train patterns: These are patterns of handwritten digits stored in a matrix of size 256 × 4649. Each digit pattern is represented
by a raster scan of 16 × 16 gray level pixel intensities normalized to [−1, 1].
2. Train labels: This matrix is of size 10 × 4649 and contains the classification labels for the digits in the training set. It provides
the true information about the digit images.
Similarly, the test set also comprises of two components:
1. Test patterns: Like the training set, this matrix is of size 256 × 4649 and contains patterns of handwritten digits. Each digit
pattern is represented by a raster scan of 16 × 16 gray level pixel intensities normalized to [−1, 1].
2. Test labels: This matrix, sized 10 × 4649, contains the classification labels for the digits in the test set, providing the true
information about the digit images. error is minimized.

## Conclusion
In our digital age, image and pattern recognition hold crucial importance for automating tasks that would otherwise consume human
time, such as identifying digits or faces. This project aims to compare the efficiency and accuracy of two algorithms: a basic approach
and an SVD basis algorithm. Our goal is to automate the identification of handwritten digits using USPS data. Automating this
task is valuable because while humans can recognize digits, it’s a mundane task prone to errors over time, whereas computers can
perform it swiftly and accurately.

This report assesses the effectiveness of these algorithms and concludes that the SVD approach outperforms the basic algorithm
significantly in achieving our objective. We then introduce the ”USPS.mat” database, containing handwritten digit data split into
training and test sets. We aim to assess our algorithms’ performance on these ”noisy” digits after training them on the ”clean” digits,
highlighting the typical machine learning paradigm of training on organized data before handling more disorderly datasets.
The centroid classification algorithm calculates the average digit images using the training data. It aggregates pixel intensities for
each digit (0 to 9) from the training patterns matrix and computes the mean intensity for each pixel across all instances of that digit.
This allows us to determine the typical gray level pixel intensities associated with each digit based on accurately labeled training
data. The SVD classification algorithm involves decomposing the training data into its singular vectors. These singular vectors
capture the essential features of the data. Then, for each test instance, the algorithm calculates expansion coefficients with respect
to these singular vectors. By comparing these coefficients, the algorithm assigns the test instance to the class with the most similar
representation, enabling classification.

In our analysis, we find that the basic algorithm struggles most with distinguishing the digit 9, while identifying 2 is comparatively
easier. Conversely, the SVD basis algorithm faces challenges in identifying digit 1 due to variations in its formation, yet excels in
recognizing digit 2, which shows minimal variability. Overall, the SVD basis algorithm demonstrates significantly higher accuracy
compared to the centroid classification. With accuracies ranging from 91.18% to 94.58%, the SVD basis algorithm outperforms the
basic algorithm by over 20% in certain cases.

The report investigates the effectiveness of centroid and SVD basis algorithms in handwritten digit recognition, highlighting the
latter’s superiority in accuracy. Through meticulous analysis, the SVD basis algorithm consistently outperforms the centroid approach,
particularly evident in challenging digit classifications, showcasing its potential for practical applications in image recognition tasks
