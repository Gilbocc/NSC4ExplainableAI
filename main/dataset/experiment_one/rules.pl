classification(X,Y,Z) :- nameWithClass(Y,X), nameWithClass(Y,Z).
classification(X,Y,Z) :- nameWithClass(Y,X).
