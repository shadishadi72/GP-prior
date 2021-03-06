function [X_train,X_test] = normalise_train_test(X_train,X_test)

n_train = size(X_train,1);
%n_test = size(X_test,1);

X = normalise([X_train;X_test]);

X_train = X(1:n_train,:);
X_test = X((n_train+1):end,:);
end