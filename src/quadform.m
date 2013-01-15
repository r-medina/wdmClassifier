%{
---------------------------------------------------------------------
Function: logreg
Name: ramedina

Header comments:
  This part is really what makes the logistic regressions I do
  special. Using all possible combinations of the coordinate variables
  for some data set paints a much, much clearer picture of where the ins
  lie with respect to their coordinate space.
---------------------------------------------------------------------
%}

% quadform outputs an array which contains all the combinations of the
% seperate dimensions in the input array up to degree two.
function x_combinations =  quadform(x)

[n,k] = size(x);

x_combinations = [x,x.^2];

t = 0;
for j = 1:k-1
  for i = j+1:k
    t = t+1;
    temp(:,t) = x(:,j).*x(:,i);
  end
end

x_combinations = [x_combinations,temp];

end