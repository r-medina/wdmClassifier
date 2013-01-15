%{
---------------------------------------------------------------------
Function: logreg
Name: ramedina

Header comments:
  This function is the meat of my project. It takes 5 arguments:
  x_train,y_train,x_test,y_test,x_unkown. The x_arrays contain
  n k dimensional vectors. For x_train and x_test the classification for
  each of the n objects is known and contained in the respective
  y_array. The k values for each of the n objects represent information
  about these objects. The user can forgo including a column vector of
  ones to the begining of the x_arrays, as the function will check for
  it and append it if it's not already there. Similarly, the y_arrays
  can either be ones and zeros or ones and negative ones for the sake of
  simplicity and reusability.
  
  To understand what this function does in a mathematical sense, a sound
  understanding of what exactly a logistic regression is is
  required. The logistic regression essentially associates a weight to
  each of the k components such that when a k dimensional vector is
  passed to some function, let's call it g(z), g(z) it returns a
  probability (range = (0,1)) for that object to be in or out based on the training
  data.

  The aforementioned "weights" are known as "beta" values and there are
  k+1 of these (as there is an "intercept"). The argument "z" that got
  passed to the function g is the dot product of the beta values and the
  k+1 dimensional vector for some object (where the +1 is a column
  vector of ones). That is, once the logistic regression is done, to
  find the probability that some k=2 object is "in" would be: 
  g(beta_1 + beta_2 * k_1 + beta_3 * k_2). g(z) is a sigmoid function:
  1/(1+exp(-z).

  How exactly the logistic regression finds these betas will be
  explained in later comments.

---------------------------------------------------------------------
%}

% The function outputs beta values, a confusion matrix for the testing
% data and arrays with information about the test and unkown data. If no
% testing data is passed, the function uses the training data and tests
% on that. If no unkown data is passed, the function still runs.
% The confusion matrix contains:
% [true positives, false positives; false negatives, true negatives].
% results_test's first column is 1 if the prediction was right and -1
% if it was wrong, the second column is a measure of correctness, the
% third column is the prediction, the fourth column
% is the probability, and the input data. results_unkown is
% similar except it's missing that first right/wrong column and the
% correctness, as that is not known.
function [Betas,confusion_matrix,results_test,results_unknown] = logreg(x_train,y_train,x_test,y_test,x_unknown)

% The process functions take the user's input and formats it
% appropriately.
x_train = process_x(x_train);
y_train = process_y(y_train);

% If no testing data, logistic regression is tested on training data.
if isempty(x_test)
  x_test = x_train;
  y_test = y_train;
else
  x_test = process_x(x_test);
  y_test = process_y(y_test);
end

% Dimensions become useful.
[n_train,k_train] = size(x_train);
[n_test,k_test] = size(x_test);

Betas = train(x_train,y_train);
test_probs = probability(Betas,x_test);
io_test = in_or_out(test_probs);
test_correctness = asses_test_predic(io_test,y_test);
results_test = [io_test.*y_test,io_test,test_probs,x_test];
confusion_matrix = test(io_test,y_test);

% unknown_results only gets set if there is an x_unknown input. 
if any(x_unknown)
  x_unknown = process_x(x_unknown);
  unknown_probs = probability(Betas,x_unknown);
  io_unknown = in_or_out(unknown_probs);
  results_unknown = [io_unknown,unknown_probs,x_unknown];
end

% Adds column vector of ones to x_arrays.
function new_x = process_x(x)
  first_col = ones(size(x,1),1);
  if any(x(:,1) ~= first_col)
    new_x = [first_col,x];
  else 
    new_x = x;
  end
end

% Makes sure y_arrays are 1 and -1 only.
function new_y = process_y(y)
  if any(y==0)
    new_y = ((y==0)*-1)+y;
  else
    new_y = y;
  end
end

% Sigmpoid function used extensively.
function output  = sigmoid(z)
  output = 1.0./(1.0+exp(-z));
end

% This is the logisticregression. train finds the beta values by
% minimizing the negative sum of log likelihoods for beta values over
% all the training data.
function betas = train(x,y)

  % Arbitrarily setting betas.
  betas = zeros(1,k_train);
  
  % Defines the negative sum of the log liklihoods and the gradient with
  % respect to each of the beta values.
  function [logL,dBeta] = negLogLike(x,y,betas)
    % Instantiating variable.
    logL = 0;
     
    % Sum of the log likelihoods.
    for i = 1:n_train
      logL = logL + log(sigmoid(y(i)*dot(betas,x(i,:))));
    end

    % Makes it negative.
    logL = -logL;

    % Defines the gradient only if the function is asked for more than
    % one output.
    if nargout > 1;
      dBeta = 0;
      % This gradient is kind of rediculous. I can't really explain
      % it. I differentiated on paper and then modelled it
      % here. Essentially you differentiate the sums over all the data
      % for each beta value. This took a very long time to get.
      for j = 1:k_train
	dBeta_k = 0;
	for i = 1:n_train
	  dBeta_k = dBeta_k - y(i)*x(i,j) * ...
	      sigmoid(-y(i)*dot(betas,x(i,:)));
	end
        % Sets the gradient as an array of all the partial derrivatives
	% with respect to each beta value.
	dBeta(j) = dBeta_k;
      end
    end
  end

  % Minimizes negLogLike with respect beta values given the gradient.   
  betas = fminunc(@(betas) negLogLike(x,y,betas),betas,optimset('GradObj','on','TolX',1e-8));
end

% Finds the probability, given beta values and some x_array, of objects
% being in.
function probabilities = probability(betas,x)  
  for i = 1:size(x,1)
    probabilities(i) = sigmoid(dot(betas,x(i,:)));
  end
  probabilities = probabilities';
end

% Says whether a probability is in or out.
function in_out = in_or_out(prob)
  in_out = prob > .5;
  in_out = (in_out == 0)*-1 + in_out;
end

% Assesses percentage of correctness for the test predictions.
function correctness = asses_test_predic(predic,y)
  correctness = (predic .* y + 1)*.5;
end

% Makes confusion matrix:
% [true positives, false positives; true negatives, false negatives].
function conf_matrix = test(inout_predic,y)
  conf_matrix = zeros(2);

  ins = y==1;
  outs = y==-1;
  ins_predic = inout_predic==1;
  outs_predic = inout_predic==-1;

  true_ins = sum(ins);
  true_outs = sum(outs);
  predic_ins = sum(ins_predic);
  predic_outs = sum(outs_predic);
  true_pos = sum((ins==1)&(ins_predic==1));
  false_pos = sum((ins==0)&(ins_predic==1));
  true_neg = sum((outs==1)&(outs_predic==1));
  false_neg = sum((outs==0)&(outs_predic==1));

  conf_matrix(1,:) = [true_pos,false_pos];
  conf_matrix(2,:) = [false_neg,true_neg];
end

end