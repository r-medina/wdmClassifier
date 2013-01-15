%{
---------------------------------------------------------------------
Function classifybymag
Name: ramedina

Header comments:
  This function works by taking 2 .csv files, train_in and train_out,
  which contain u,g,r,i,z magnitudes and running a logistic
  regression. The train_in can be classified as anything, but it has to
  be a binary classification and there should be some confidence as to
  the absence of accidental overlap between train_in and train_out This
  means that if specific types of stars do occupy discrete bundles in
  color-color space, that this code is rubust and can be used to
  identify where precicely these bundles are.

  Unfortunately, for the stars in question for this project, white
  dwarf-m dwarf binaries, it is still inconclusive as to whether this
  model works incredibly well. I did the logistic regression with 520
  known ins from Silvestri's catalague and 11 thousand "known" outs from
  the same data release. Everything runs smoothly, but the confusion
  matrix is giving me 341 true positives, 104 false positives, 179 false
  negatives and 10896 true negatives (I ran the training data back into the
  probability sigmoid). This is not completely abismal since out of 11
  thousand it's only guessing incorrectly for 179. I'm ok with that.

  Logistic regressions like this (with combining the coordinate
  variables into a quadratic form) are awesome. With a 2 dimensional
  data set, if the "in" data lies in some clump, the probability surface
  with respect to the coordinate plane actually reflects where that data
  is. That is, if the z axis is probability and the x and y plane are
  the two pieces of data, there will be a very well defined ellipse or
  other conic section shape with a height of one over the clump of your
  in data. When I run this on the steller data (which gives a 14-term quadratic
  form), the 3 dimensional representations of that 4D quadratic form
  isn't always a sphere, but it does show where the WDM binaries are for
  that combination of colors. The problem, then, I think, is either that
  my hypothesis was wrong (WDM binaries aren't the only stars that
  take up that cluster in color-color space), or that the data I'm using as
  "outs" actually has tons of WDM binaries in it. I grep-ed the known
  ins against 11 thousand stars from the same data release that
  Silvestri extracted her catalogue from and got 0 overlap. I checked
  the regular expression I was using and tested it a million ways and
  then checked it and checked it again, but I didn't find a single star
  out of 11 thousand that I got at random to be in 520 of Silvestri's
  WDM binaries. Given that there are 850 thousand items with spectra in
  DR4 and Silvestri only classified 520 that were very certain or not
  already catalogued by Raymond, I suppose it is possible that none of
  those 11 thousand were WDM binaries. That means that you would have to
  put an insanely large amount of stars in to hope to find ANY WDM
  binaries. Sad.

  Lowering the size of the "out" data to 4 thousand as opposed to 11
  actually helps the true positive count considerably (but increases the
  false positive to true negative ratio a lil) since that eliminates a lot
  of the overlap during the regression. If the overlap is coincidental, having a set of 11
  thousand will emphasize that overlap significantly and obfuscate the
  result.

  The surfaces that this function graphs are confusing. The surfaces
  seem to get no where near where the stars are, however, if you
  consider that the equation of the surface is 0 = betas dot x_vectors,
  then, in theory, everything outside that surface is an "out" (because
  the sigmoid is 1/(1+exp(-z)) and on the surface would be -z = 0
  therefore a probability of 1/2). In my graphs, the definition of "in"
  and "out" is either not comprehensible in 3d (as the surface never
  even intercepts the data) or thereis some flaw in the way I'm graphing
  them.

  In any case, it more or less works. Depending on what the user inputs,
  the code can be altered to make it run faster and to play around with
  values to see the effect on the confusion matrix returned by the
  logistic regression. This function not only returns the betas, but it
  saves them to a .csv. The csvtitle argument should contain the name of
  what you're classifying as well as the iteration you're on (so you
  don't overwrite old ones)--this is just for self reference.
---------------------------------------------------------------------
%}

% Depending on the combination of arguments passed, the function does
% several things. At the very least, the user has to provide two .csv
% files containing "ins" and "outs."
% The betas are always returned and saved into a .csv. Another thing
% that is always returned and saved is a results matrix (as described in
% logreg.m) for the testing data (if no testing data is give, then the
% training data will be tested). If unknown data is given, the function
% will aslo save its results matrix.
function classifybymag(train_in,train_out,test_in,test_out,unkown_stars,csvtitle)

% Formats the magnitude data into proper color-color data and saves how
% many ins there are for use in later commands.
[X_train,Y_train,num_ins] = makexy(train_in,train_out);
n_total = size(X_train,1);
% Formats X_train into a quadratic form to then pass to the logistic
% regression.
X_train = quadform(X_train);

if ~isempty(test_in) & ~isempty(test_out)
  [X_test,Y_test] = makexy(test_in,test_out)
elseif (isempty(test_in)+isempty(test_out) == 1)
  disp('Error: one of the testing files wasn''t found.')
  return;
else
  X_test = [];
  Y_test = [];
end

if ~isempty(unkown_stars)
  X_unknown = color_color(unknown_stars)
else
  X_unknown = [];
end

% The four graphs made here are the 4 3 dimensional projections of the 4
% dimensional color-color space:
% (u-g,g-r,r-i), (g-r,r-i,i-z), (u-g,r-i,i-z), and (u-g,g-r,i-z).
close all;
af = figure;
scatter3(X_train(1:num_ins,1),X_train(1:num_ins,2),X_train(1:num_ins,3),6,'k','filled');
hold on;
scatter3(X_train(num_ins+1:n_total,1),X_train(num_ins+1:n_total,2),X_train(num_ins+1:n_total,3),.3,'r','filled');
axis([-4 4 -4 4 -4 4]);
title('u-g vs. g-r vs. r-i');
%saveas(af,'../graphs/uggrri.png');

bf = figure;
scatter3(X_train(1:num_ins,2),X_train(1:num_ins,3),X_train(1:num_ins,4),6,'k','filled');
hold on;
scatter3(X_train(num_ins+1:n_total,2),X_train(num_ins+1:n_total,3),X_train(num_ins+1:n_total,4),.3,'r','filled');
axis([-4 4 -4 4 -4 4]);
title('g-r vs. r-i vs. i-z')
%saveas(bf,'../graphs/grriiz.png');

cf = figure;
scatter3(X_train(1:num_ins,1),X_train(1:num_ins,3),X_train(1:num_ins,4),6,'k','filled');
hold on;
scatter3(X_train(num_ins+1:n_total,1),X_train(num_ins+1:n_total,3),X_train(num_ins+1:n_total,4),.3,'r','filled');
axis([-4 4 -4 4 -4 4]);
title('u-g vs. r-i vs. i-z');
%saveas(cf,'../graphs/ugriiz.png');

df = figure;
scatter3(X_train(1:num_ins,1),X_train(1:num_ins,2),X_train(1:num_ins,4),6,'k','filled');
hold on;
scatter3(X_train(num_ins+1:n_total,1),X_train(num_ins+1:n_total,3),X_train(num_ins+1:n_total,4),.3,'r','filled');
axis([-4 4 -4 4 -4 4]);
title('u-g vs. g-r vs. i-z');
%saveas(df,'../graphs/uggriz.png');

%return;

%[betas,conf_matrix] = logreg(X_train,Y_train,X_test,Y_test,X_unknown)

%if (isempty(test_in) & isempty(unkown_stars))
%  [betas,conf_matrix,test_res,unkown_res] = logreg(X_train,Y_train,[],[],[])
%elseif (~isempty(test_in) & isempty(unkown_stars))
%  [betas,conf_matrix,test_res,unkown_res] = logreg(X_train,Y_train,X_test,Y_test,[])
%elseif (isempty(test_in) & ~isempty(unkown_stars))
%  [betas,conf_matrix,test_res,unkown_res] = logreg(X_train,Y_train,[],[],X_unkown)
%elseif ~(isempty(test_in) | isempty(unkown_stars))
%  [betas,conf_matrix,test_res,unkown_res] = logreg(X_train,Y_train,X_test,Y_test,X_unknown)
%end

if ~isempty(X_unknown)
  [betas,conf_matrix,test_res,unkown_res] = logreg(X_train,Y_train,X_test,Y_test,X_unknown);
  csvwrite(sprintf('../output/%s_unknown.csv',csvtitle),unknown_res);
else
  [betas,conf_matrix,test_res] = logreg(X_train,Y_train,X_test,Y_test,X_unknown);
end

csvwrite(sprintf('../output/%s_betas.csv',csvtitle),betas);
csvwrite(sprintf('../output/%s_testresults.csv',csvtitle),test_res);

betas
conf_matrix

% Adds 3d projections of the 4d the surfaces made by the quadratic forms
% and three color-color dimensions.
conicproj3(betas([1 2 3 4 6 7 8 10 11 13]),af)
conicproj3(betas([1 3 4 5 7 8 9 13 14 15]),bf)
conicproj3(betas([1 2 4 5 6 8 9 11 12 15]),cf)
conicproj3(betas([1 2 3 5 6 7 9 10 12 14]),df)

% Subtracts contiguous color bands from eachother to get the color-color
% values.
function color_x = color_color(x)
  color_x = [x(:,1)-x(:,2),x(:,2)-x(:,3),x(:,3)-x(:,4),x(:,4)-x(:,5)];
end

% Reads the .csv file arguments for training and testing purposes and
% makes them arrays. x contains the data and y contains the classification.
function [x,y,n_in] = makexy(ins,outs)
  in = csvread(ins);
  out = csvread(outs);
  %out = out(200062500,:);
  n_in = size(in,1);
  n_out = size(out,1);
  x = [in;out];
  x = color_color(x);
  y = [ones(n_in,1);zeros(n_out,1)];
end

% Function takes an array of ten beta values and a reference to the
% corresponding figure and makes the graph of the surface corresponding
% to that space.
function conicproj3(beta_vals,whichone)
  inter=beta_vals(1);
  a=beta_vals(2);
  b=beta_vals(3);
  c=beta_vals(4);
  a2=beta_vals(5);
  b2=beta_vals(6);
  c2=beta_vals(7);
  ab=beta_vals(8);
  ac=beta_vals(9);
  bc=beta_vals(10);

  % Makes the mesh grid to pass to the implicit surface equation which
  % is a function of three color bands combined into a quadratic form
  % and multiplied by the appropriate beta values.
  r = linspace(-4,4,15);
  [s,t,u] = meshgrid(r,r,r);

  % This is the equation for any 3d projection of the 4d surface that
  % corresponds to the 15 beta values that the logistic regression returns.
  P=inter+a*s+b*t+c*u+a2*s.^2+b2*t.^2+c2*u.^2+ ...
      ab*s.*t+bc*t.*u+ac*s.*u;

  figure(whichone);  
  p=patch(isosurface(s,t,u,P));
  isonormals(s,t,u,P,p)
  set(p,'FaceColor','b','EdgeColor','k','FaceAlpha',0.5);
  lighting gouraud;  
end

end