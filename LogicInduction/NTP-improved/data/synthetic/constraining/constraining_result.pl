classification(X,Y,Z) :- surnameWithClass(Y,X), surnameWithClass(Y,Z).
classification(X,Y,Z) :- surnameWithClass(Y,X).
