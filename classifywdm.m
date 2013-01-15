%{
---------------------------------------------------------------------
Function classifybymag
Name: ramedina

Header comments:
  This is my master function. Run this to run the program. Extensive
  header comments explain a bit of the math behind the code. Enjoy.

  It takes about half an hour to run unless you uncomment line 188 in
  classifybymag.m

  The way it works is that this function calls classifybymag which does
  a classification on any steller data given that you have the
  magnitudes for known ins and outs (only does a binary
  classification). That function calls the logreg function which runs
  the logistic regression and gets the results. classifybymag, though,
  also plots 4 3d projections of the data and then plots the surfaces
  produced by the beta values and x data.
---------------------------------------------------------------------
%}

function classifywdm

classifybymag('dr4silv.csv','dr4psfstars.csv',[],[],[],'dr4wdmbinary')

end