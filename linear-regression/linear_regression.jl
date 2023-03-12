using MLDatasets, Plots
using Statistics, Distributions
using MLJ
using Random
using DataFrames


X = rand(1:100, 100)
global a = rand(1:5, 1)[1]
global b = rand(1:5, 1)[1]

function train_linear_regression(X, y)
    """
    Trains Linear Regression on the dataset (X, y).
    
    Parameters
    ----------
    X : numpy array, shape (m, d + 1)
        The matrix of input vectors (each row corresponds to an input vector); 
        the first column of this matrix is all ones (corresponding to x_0).
    y : numpy array, shape (m, 1)
        The vector of outputs.
    
    Returns
    -------
    w : numpy array, shape (d + 1, 1)
        The vector of parameters of Linear Regression after training.
    """

    w = pinv(transpose(X) * X) * transpose(X) * y 

    
    
    return w
end


function bias_initializer(X)
    num_row, num_col = size(X,1), size(X,2)
    one_added_X = fill(1, num_row, num_col + 1)
    new_num_row, new_num_col = size(one_added_X,1), size(one_added_X,2)
    
    for i in 1:(new_num_row)
        for j in 2:(new_num_col)
            one_added_X[i, j] = X[i, j - 1]
        end 
    end
    
    return one_added_X
end

f(x) = a*x + b + rand(1:30,1)[1]
y = f.(X);
print("Your regression function: y = $a*x + $b + noise")
scatter(X, y, label="data points", xlabel="x", ylabel="y", title="Visualization of data", legend=false)


one_added_X  = bias_initializer(X)

println("one_added_X.shape =", size(one_added_X))
println("y.shape =", size(y))




w = train_linear_regression(one_added_X, y)

predicted_ys = one_added_X * w
scatter(X, y, xlabel="x", ylabel="y", title="Visualization of data", legend=false)
x_min = 0
x_max = 100
xs = [x_min x_max]'

ones_added_xs  = bias_initializer(xs)

predicted_ys = ones_added_xs*w
scatter!(xs, predicted_ys, legend=false)
plot!(xs, predicted_ys, legend=false)
