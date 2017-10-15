function [J,grad] = nnCostFunction(nn_params, ...
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
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));
Y=zeros(num_labels,1);
ytest=Y;
delta2nd=zeros(num_labels,hidden_layer_size+1);
delta1st=zeros(hidden_layer_size,input_layer_size+1);
for i=1:m
	a1=X(i,:);
	a1=[1 a1];
	a2=(sigmoid(a1*Theta1'));
	a2=[1 a2];
	a3=(sigmoid(a2*Theta2'));
	h(:,i)=a3;
	ytest(y(i))=1;
	J=J+(-1/m)*sum((ytest.*log(h(:,i)))+((1.-ytest).*log(1-h(:,i))))
	ytest=Y;
end;
theta1=Theta1(:,2:input_layer_size+1);
theta2=Theta2(:,2:hidden_layer_size+1);
for i=1:hidden_layer_size
	t1(i)=sum(theta1(i,:).^2);
end;
for j=1:num_labels
	t2(j)=sum(theta2(j,:).^2);
end;
regularization=(lambda/(2*m))*(sum(t1)+sum(t2));
J=J+regularization;

%Backpropagation and Gradient Calculation......

for i=1:m
	a1=X(i,:);
	a1=[1 a1];
	a2=sigmoid(a1*Theta1');
	a2=[1 a2];
	a3=sigmoid(a2*Theta2');
	ytest(y(i))=1;
	del3rd=a3'-ytest;
	del2nd=Theta2'*del3rd.*(sigmoidGradient([1 a1*Theta1']))';
	del2nd=del2nd(2:end);
	delta2nd=delta2nd+(del3rd*a2);
	delta1st=delta1st+(del2nd*a1);
	ytest=Y;
end;
Theta1_grad(:,1)=delta1st(:,1)/m;
Theta1_grad(:,2:input_layer_size+1)=(delta1st(:,2:input_layer_size+1)/m)+((lambda/m).*theta1);
Theta2_grad(:,1)=delta2nd(:,1)/m;
Theta2_grad(:,2:hidden_layer_size+1)=(delta2nd(:,2:hidden_layer_size+1)/m)+((lambda/m).*theta2);



























% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
