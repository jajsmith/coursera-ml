function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

[J, grad] = costFunction(theta, X, y);


% COST
S = sum(theta(2:length(theta)) .^ 2);
J = J + lambda / (2 * m) * S;

% GRADIENT
% gradient seems to be a column-wise vector so we have to transpose the regularization term
grad(2:length(grad)) = grad(2:length(grad)) + lambda / m .* theta(2:length(theta))';

end
