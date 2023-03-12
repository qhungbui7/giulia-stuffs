import Pkg; 
Pkg.add("MLJ")
Pkg.add("DataFrames")
Pkg.add("VegaLite")
Pkg.add("Distributions")
Pkg.add("Plots")

using MLJ, DataFrames, VegaLite, Distributions
using LinearAlgebra, Plots


function sigmoid_activation(x)
    tmp = 1. ./ (1. .+ exp.(-x)) 
    return tmp 

end

function sigmoid_deriv(x)
    """
    Compute the derivative of the sigmoid function ASSUMING
    that the input 'x' has already been passed through the sigmoid
    activation function
    """

    sig = x
    tmp = sig .* (1. .- sig)
    return tmp


end


function compute_h(W, X) # done

    """
    Compute output: Take the inner product between our features 'X' and the weight matrix 'W'.
    """
    tmp = X * W
    return tmp


end

function predict(W, X) # done
    """
    Take the inner product between our features and weight matrix, 
    then pass this value through our sigmoid activation
    """
    preds = sigmoid_activation(compute_h(W, X))

    # apply a step function to threshold the outputs to binary
    # class labels
    preds[preds .<= 0.5] .= 0
    preds[preds .> 0.] .= 1

    return preds
end




function compute_gradient(error, y_hat, trainX)
    """
    the gradient descent update is the dot product between our
    features and the error of the sigmoid derivative of
    our predictions
    """

    grad = transpose(X_train) * (error .* sigmoid_deriv(y_hat))

    return  grad
end

function train(W, trainX, trainY, learning_rate, num_epochs)
    losses = []
    for epoch in 1:num_epochs
        y_hat = sigmoid_activation(compute_h(W, trainX))

        error = y_hat - trainY

        append!(losses, 0.5 * sum(error .^ 2)) 

        grad = compute_gradient(error, y_hat, trainX)
        W -= learning_rate * grad # 1 x 3

        if epoch == 1 || epoch % 5 == 0
            println("Epoch=$epoch; Loss=$(losses[end])")
        end
    end
    return W, losses
end



# generate a 2-class classification problem with 1,000 data points, each data point is a 2D feature vector
X, y = make_blobs(1000, 2, centers=2, cluster_std=0.5, rng=1)
df = DataFrame(X)
df.y = convert(Vector{Float64}, y) .- 1

# insert a column of 1’s as the last entry in the feature matrix  
# -- allows us to treat the bias as a trainable parameter
df.x3 = ones(size(df)[1],)

# Split data, use 50% of the data for training and the remaining 50% for testing
df_train, df_test = partition(df, 0.5)
X_train, y_train = [df_train.x1 df_train.x2 df_train.x3], df_train.y
X_test, y_test = [df_test.x1 df_test.x2 df_test.x3], df_test.y;



df |> @vlplot(
    :point, 
    x=:x1, y=:x2, 
    color = :"y:n",
    width=400,height=400
)

W = rand(Normal(), (size(X_train)[2], 1))
print(size(compute_h(W, X_train)))

num_epochs= 100
learning_rate= 0.1
W, losses = train(W, X_train, y_train, learning_rate, num_epochs)
plot(1:num_epochs, losses, legend=false)


preds = predict(W, X_test)
acc = accuracy(preds, reshape(y_test, length(y_test), 1))

p = precision(preds, reshape(y_test, length(y_test), 1))
r = recall(preds, reshape(y_test, length(y_test), 1))
f1 = 2*p*r/(p + r)
print("acc: $acc, precision: $p, recall: $r, f1_score: $f1\n")


# visualize the result of predictions
df_test.y_hat = reshape(preds, (length(preds),))
df_test |> @vlplot(
    :point, 
    x=:x1, y=:x2, 
    color = :"y_hat:n",
    width=400,height=400
)