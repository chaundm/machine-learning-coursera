import numpy as np
import sigmoid as s
import sigmoidGradient as sg

def nnCostFunction(nn_params, input_layer_size, hidden_layer_size, \
	num_labels, X, y, lambda_reg):
    #NNCOSTFUNCTION Implements the neural network cost function for a two layer
    #neural network which performs classification
    #   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
    #   X, y, lambda) computes the cost and gradient of the neural network. The
    #   parameters for the neural network are "unrolled" into the vector
    #   nn_params and need to be converted back into the weight matrices.
    #
    #   The returned parameter grad should be a "unrolled" vector of the
    #   partial derivatives of the neural network.


    # Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
    # for our 2 layer neural network
	Theta1 = np.reshape(nn_params[:hidden_layer_size * (input_layer_size + 1)], \
                     (hidden_layer_size, input_layer_size + 1), order='F')

	Theta2 = np.reshape(nn_params[hidden_layer_size * (input_layer_size + 1):], \
                     (num_labels, hidden_layer_size + 1), order='F')

    # Setup some useful variables
	m = len(X)

    # # You need to return the following variables correctly
	J = 0;
	Theta1_grad = np.zeros( Theta1.shape )
	Theta2_grad = np.zeros( Theta2.shape )

    # ====================== YOUR CODE HERE ======================
    # Instructions: You should complete the code by working through the
    #               following parts.
    #
    # Part 1: Feedforward the neural network and return the cost in the
    #         variable J. After implementing Part 1, you can verify that your
    #         cost function computation is correct by verifying the cost
    #         computed in ex4.m
    #
    # Part 2: Implement the backpropagation algorithm to compute the gradients
    #         Theta1_grad and Theta2_grad. You should return the partial derivatives of
    #         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
    #         Theta2_grad, respectively. After implementing Part 2, you can check
    #         that your implementation is correct by running checkNNGradients
    #
    #         Note: The vector y passed into the function is a vector of labels
    #               containing values from 1..K. You need to map this vector into a
    #               binary vector of 1's and 0's to be used with the neural network
    #               cost function.
    #
    #         Hint: We recommend implementing backpropagation using a for-loop
    #               over the training examples if you are implementing it for the
    #               first time.
    #
    # Part 3: Implement regularization with the cost function and gradients.
    #
    #         Hint: You can implement this around the code for
    #               backpropagation. That is, you can compute the gradients for
    #               the regularization separately and then add them to Theta1_grad
    #               and Theta2_grad from Part 2.
    #

    # PART 1 - NONREGULARIZED COST FUNCTION
	X = np.column_stack((np.ones((m,1)), X)) #X = m x (n+1); Theta1 = L x (n+1)
	a1 = X #a1 = m x (n+1)

	z2 = a1.dot(Theta1.T) #z2 = m x L
	a2 = s.sigmoid( z2 ) #a2 = m x L
	a2 = np.column_stack((np.ones((a2.shape[0],1)), a2)) #a2 = m x (L+1); Theta2 = K x (L+1)

	z3 = a2.dot(Theta2.T)
	a3 = s.sigmoid( z3 ) #a3 = m x K

	labels = y # recode labels as vectors containing only values 0 or 1
	y = np.zeros((m,num_labels)) # set y to be matrix of size m x k
	for i in range(m):
		y[i, labels[i]-1] = 1

	#Two sum function are the trace of matrix
	J = 1/m*np.trace(-y.T.dot(np.log(a3)) - (1-y.T).dot(np.log(1-a3)));

    # PART 2 - REGULARIZED COST FUNCTION
    # note that Theta1[:,1:] is necessary given that the first column corresponds to transitions
    # from the bias terms, and we are not regularizing those parameters. Thus, we get rid
    # of the first column.

	sumOfTheta1 = np.sum(np.sum(Theta1[:,1:]**2))
	sumOfTheta2 = np.sum(np.sum(Theta2[:,1:]**2))

	J = J + lambda_reg/(2.0*m)*(sumOfTheta1+sumOfTheta2)

    # PART 3 - BACK PROPAGATION

	# %Step1
	# %Done
	# %Step2
	delta3 = a3 - y # delta3 = m x K
	# %Step3
	delta2 = delta3.dot(Theta2[:,1:])*sg.sigmoidGradient(z2) #elta2 = m x L
	# %Step4
	Delta1 = delta2.T.dot(a1) #Delta1 L x (n+1)
	Delta2 = delta3.T.dot(a2) #Delta2 K x (L+1)
	# %step5
	Theta1_grad = 1/m*Delta1
	Theta1_grad[:,1:] += lambda_reg/m*Theta1[:,1:]
	Theta2_grad = 1/m*Delta2
	Theta2_grad[:,1:] += lambda_reg/m*Theta2[:,1:]

	# -------------------------------------------------------------
    # # =========================================================================

    # Unroll gradients
	grad = np.concatenate((Theta1_grad.reshape(Theta1_grad.size, order='F'),
							Theta2_grad.reshape(Theta2_grad.size, order='F')))

	return J, grad
