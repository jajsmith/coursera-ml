function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
K = num_labels;
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%

%size(X)
%size(y)
%size(Theta1)
%size(Theta2)
%Theta2
%fprintf('Program paused. Press enter to continue.\n');
%pause;

% Start with iterative (Rather than vectorized) version
sum_samples = 0;
for i=1:m;
  sum_k = 0;
  sample_i = X(i,:)'; % vector of inputs taken from X
  a1_i = [1; sample_i]; % Add bias 

  % Convert label to n*1 matrix 
  sample_i_label = y(i); % label for sample
  sample_i_label_vec = zeros(num_labels,1);
  sample_i_label_vec(sample_i_label) = 1;

  % Calculate hypothesis from Neural Net
  a2_i = [1; sigmoid( Theta1 * a1_i )]; % activation of kth output unit
  a3_i = sigmoid( Theta2 * a2_i );
  hypothesis = a3_i;

  % Cost of sample
  sum_k = sum( - sample_i_label_vec .* log( hypothesis ) ...
                 - (1 - sample_i_label_vec) .* log( 1 - hypothesis) ...
               );
  %fprintf('cost for label at sample\n');
  %i
  %sample_i_label
  %[sample_i_label_vec, hypothesis]
  %sum_k
  %pause;
  sum_samples += sum_k;
end;
%fprintf('Calculated sums over i/m/K\n');
%i
%m
%K
J = sum_samples / m;

% Regularization
Theta1_nobias = Theta1(:, 2:size(Theta1)(2));
Theta2_nobias = Theta2(:, 2:size(Theta2)(2));
sum_theta1 = sum( sum( Theta1_nobias .^ 2 ) );
sum_theta2 = sum( sum( Theta2_nobias .^ 2 ) );
regularization_term = lambda / (2*m) * (sum_theta1 + sum_theta2);
J = J + regularization_term;

% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.


% First compute forward pass (all activation values and hypothesis
for t=1:m;
  % For each sample t
  sample_t = X(t,:)'; % vector of inputs taken from X

  % Convert label to n*1 matrix 
  sample_i_label = y(t); % label for sample
  sample_i_label_vec = zeros(num_labels,1);
  sample_i_label_vec(sample_i_label) = 1;

  % Calculate hypothesis from Neural Net
  % This is the forward propogation step
  a1 = [1; sample_t]; % 1 is to add bias 
  z2 = Theta1 * a1;
  a2 = [1; sigmoid( z2 )]; % activation of kth output unit
  z3 = Theta2 * a2;
  a3 = sigmoid( z3 );
  hypothesis = a3;


  delta3 = a3 - sample_i_label_vec;
  delta2_part = Theta2' * delta3;
  delta2 = delta2_part(2:end) .* sigmoidGradient( z2 );

  Theta1_grad = Theta1_grad + delta2 * a1';
  Theta2_grad = Theta2_grad + delta3 * a2';
end;

% Regularize gradients
Theta1_grad = Theta1_grad / m;
Theta2_grad = Theta2_grad / m;

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
