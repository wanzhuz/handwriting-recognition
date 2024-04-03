"""
Wanzhu Zheng
ID: 918166670
"""

from scipy import io
import matplotlib.pyplot as plt
import numpy as np

"""
Step 01: Download USPS.mat and plot the first 16 images
"""
data_dict = io.loadmat('USPS.mat')  # download USPS.mat
train_patterns = data_dict["train_patterns"]  # store train_patterns data
train_p2dArray = np.array(train_patterns)  # put train_patterns data into an array

fig = plt.figure(figsize=(10, 7))  # initialize a figure 10in wide, 7 in tall
rows = 4  # figure will have 4 rows
columns = 4  # and 4 columns

# get data for first 16 images
img1 = np.reshape(train_p2dArray[:, 0], (16, 16))
img2 = np.reshape(train_p2dArray[:, 1], (16, 16))
img3 = np.reshape(train_p2dArray[:, 2], (16, 16))
img4 = np.reshape(train_p2dArray[:, 3], (16, 16))
img5 = np.reshape(train_p2dArray[:, 4], (16, 16))
img6 = np.reshape(train_p2dArray[:, 5], (16, 16))
img7 = np.reshape(train_p2dArray[:, 6], (16, 16))
img8 = np.reshape(train_p2dArray[:, 7], (16, 16))
img9 = np.reshape(train_p2dArray[:, 8], (16, 16))
img10 = np.reshape(train_p2dArray[:, 9], (16, 16))
img11 = np.reshape(train_p2dArray[:, 10], (16, 16))
img12 = np.reshape(train_p2dArray[:, 11], (16, 16))
img13 = np.reshape(train_p2dArray[:, 12], (16, 16))
img14 = np.reshape(train_p2dArray[:, 13], (16, 16))
img15 = np.reshape(train_p2dArray[:, 14], (16, 16))
img16 = np.reshape(train_p2dArray[:, 15], (16, 16))

# for each img, create a subplot to the figure and plot the image in their corresponding subplot
fig.add_subplot(rows, columns, 1)
plt.imshow(img1, cmap='gray')
plt.axis('off')
plt.title("Image 1")

fig.add_subplot(rows, columns, 2)
plt.imshow(img2, cmap='gray')
plt.axis('off')
plt.title("Image 2")

fig.add_subplot(rows, columns, 3)
plt.imshow(img3, cmap='gray')
plt.axis('off')
plt.title("Image 3")

fig.add_subplot(rows, columns, 4)
plt.imshow(img4, cmap='gray')
plt.axis('off')
plt.title("Image 4")

fig.add_subplot(rows, columns, 5)
plt.imshow(img5, cmap='gray')
plt.axis('off')
plt.title("Image 5")

fig.add_subplot(rows, columns, 6)
plt.imshow(img6, cmap='gray')
plt.axis('off')
plt.title("Image 6")

fig.add_subplot(rows, columns, 7)
plt.imshow(img7, cmap='gray')
plt.axis('off')
plt.title("Image 7")

fig.add_subplot(rows, columns, 8)
plt.imshow(img8, cmap='gray')
plt.axis('off')
plt.title("Image 8")

fig.add_subplot(rows, columns, 9)
plt.imshow(img9, cmap='gray')
plt.axis('off')
plt.title("Image 9")

fig.add_subplot(rows, columns, 10)
plt.imshow(img10, cmap='gray')
plt.axis('off')
plt.title("Image 10")

fig.add_subplot(rows, columns, 11)
plt.imshow(img11, cmap='gray')
plt.axis('off')
plt.title("Image 11")

fig.add_subplot(rows, columns, 12)
plt.imshow(img12, cmap='gray')
plt.axis('off')
plt.title("Image 12")

fig.add_subplot(rows, columns, 13)
plt.imshow(img13, cmap='gray')
plt.axis('off')
plt.title("Image 13")

fig.add_subplot(rows, columns, 14)
plt.imshow(img14, cmap='gray')
plt.axis('off')
plt.title("Image 14")

fig.add_subplot(rows, columns, 15)
plt.imshow(img15, cmap='gray')
plt.axis('off')
plt.title("Image 15")

fig.add_subplot(rows, columns, 16)
plt.imshow(img16, cmap='gray')
plt.axis('off')
plt.title("Image 16")

plt.show()

"""
Step 02: Compute the mean digits in the train_patterns and put them in a matrix.
         Display the 10 mean digit images
"""
train_labels = data_dict["train_labels"]  # get train_labels data
train_aves = np.zeros((256, 10))  # initialize train_aves matrix
for k in range(10):
    digit_patterns = train_patterns[:, np.where(train_labels[k, :] == 1)[0]]  # select patterns for digit k
    mean_digit = np.mean(digit_patterns, axis=1)  # compute mean digit for k
    train_aves[:, k] = mean_digit  # store mean_digit into train_aves matrix
    plt.subplot(2, 5, k + 1)  # create subplots

    # plot the mean_digit image
    plt.imshow(np.reshape(mean_digit, (16, 16)), cmap='gray')
    plt.title('Mean Digit ' + str(k))
    plt.axis('off')

plt.show()

"""
Step 03: Conduct the simplest classification
"""
# (a)
test_patterns = data_dict["test_patterns"]  # store test_patterns data
test_classif = np.zeros((10, 4649))  # initialize test_classif matrix
for k in range(10):
    # compute squared Euclidean distances between test patterns and kth mean digit
    # distances = np.sum((test_patterns - np.tile(train_aves[:, k], (1, 4649))) ** 2)
    distances = np.sum((test_patterns - train_aves[:, k][:, np.newaxis]) ** 2, axis=0)
    test_classif[k, :] = distances  # store distances in test_classif matrix

# (b)
test_classif_res = [0] * len(test_classif[0])  # initialize test_classif_res list
# goal: find the position index of the minimum of each column of test_classif
for j in range(len(test_classif[0])):
    tmp = np.min(test_classif[:, j])
    ind = np.argmin(test_classif[:, j])
    test_classif_res[j] = ind  # return index in the jth position of matrix

# (c)
test_confusion = np.zeros((10, 10))  # initialize confusion matrix
test_labels = data_dict["test_labels"]  # store test_labels data
arr = np.array(test_classif_res)  # convert list to numpy array

# goal: gather the classification results for each digit
for k in range(10):
    tmp = arr[test_labels[k, :] == 1]  # get the classification results for the kth digit
    # count the occurrences of each result
    for j in range(10):
        test_confusion[k, j] = np.count_nonzero(tmp == j)

print("Confusion Matrix:")
print(test_confusion)  # print our confusion matrix

"""
Step 04: Conduct SVD-based classification computation
"""
# (a)
train_u = np.zeros((256, 17, 10))  # initialize matrix to 256x17x10
for k in range(10):
    tmp, tmp1, tmp2 = np.linalg.svd(train_patterns[:, train_labels[k, :] == 1])  # find SVD and store into 3 matrices
    train_u[:, :, k] = tmp[:, :17]  # store left singular vectors of the kth digit into the vector

# (b)
test_svd17 = np.zeros((17, 4649, 10))  # initialize matrix to 17x4649x10
for k in range(10):
    test_svd17[:, :, k] = train_u[:, :, k].T @ test_patterns  # compute 17 × 10 numbers for each test digit image
# print(test_patterns.shape) 256x4649


# (c)
# compute the error between each original test digit image
# and its rank 17 approximation using the kth digit images in the training dataset
test_svd17res = np.zeros((10, 4649))  # initialize matrix to 10x4649
for k in range(10):
    approx = train_u[:, :, k] @ test_svd17[:, :, k]  # find rank 17 approximation of test digits (256x4649)
    error = np.linalg.norm(test_patterns - approx, axis=0)  # calculate error between original test digit image and its rank 17 approximation
    test_svd17res[k, :] = error  # store error in matrix

# (d)
# compute the confusion matrix using the SVD-based classification method
test_svd17_confusion = np.zeros((10, 10))  # initialize the confusion matrix
test_classif_svd = np.argmin(test_svd17res, axis=0)  # find the position index of the minimum of each column

for kk in range(10):
    tmp_svd = test_classif_svd[test_labels[kk, :] == 1]  # get the classification results for the kkth digit
    for jj in range(10):
        test_svd17_confusion[kk, jj] = np.count_nonzero(tmp_svd == jj)  # count frequencies of "correct" and put them in confusion matrix

print("Confusion Matrix:")
print(test_svd17_confusion)  # print the confusion matrix


