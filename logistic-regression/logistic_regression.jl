import Pkg; Pkg.add("Flux")
import Pkg; Pkg.add("Images")
import Pkg; Pkg.add("MLDatasets")
import Pkg; Pkg.add("Plots")

import Pkg; Pkg.add("Statistics")
import Pkg; Pkg.add("Distributions")

import Pkg; Pkg.add("MLJ")


using Flux, Images, MLDatasets, Plots
using Statistics, Distributions
using MLJ 
gr()


function compute_average_intensity(x, y)
    mean = zeros(10) # 10 is number of labels
    cnt = zeros(10) # 10 is number of labels
    
    num_samp = size(x)[1]
    num_feat = size(x)[2]

    for i in 1:num_samp
        mean[y[i] + 1] += Statistics.sum(x[i, :])
        cnt[y[i] + 1] += 1
    end
    # mean = Statistics.mean(x, dims=2)
    return mean ./ cnt
end

function compute_average_intensity_each_sample(x, y)
    num_samp = size(x)[1]
    num_feat = size(x)[2]
    mean = zeros(num_samp)


    for i in 1:num_samp
        mean[i] += Statistics.mean(x[i, 1:num_feat])
    end
    return mean # ./ num_samp
end

function compute_symmetry(train_x)
    symmetry = []
    for i in 1:size(train_x)[1]
        img = reshape(train_x[i,:], (28,28))
        s1 = mean(abs.(img - reverse(img, dims=1)))
        s2 = mean(abs.(img - reverse(img, dims=2)))
        s = -0.5 .* (s1 + s2)
        append!(symmetry, s)
    end
    return symmetry
end

function normalize(x_train, mean_=nothing, std_=nothing)
    if mean_ === nothing && std_ === nothing
        num_samp = size(x_train)[1]
        num_feat = size(x_train)[2]

        mean_ = Statistics.mean(x_train[:, 1:num_feat], dims=1)
        std_ = Statistics.std(x_train[:, 1:num_feat], dims=1)


        normalized_train_x = (x_train .- mean_) ./ std_  
        

        return normalized_train_x, mean_, std_
    end
    
    normalized_train_x = (x_train .- mean_) ./ std_  

    return normalized_train_x # return `normalized_train_x` calculated by using mean_ and std_ (both of them are passed and not null)
end

function sigmoid_activation(x)
    """
    Compute the sigmoid activation value for a given input
    """
    return 1.0 ./ (1.0 .+ exp.(-x))
end

function sigmoid_deriv(x)
    """
    Compute the derivative of the sigmoid function ASSUMING
    that the input 'x' has already been passed through the sigmoid
    activation function
    """
    return x .* (1 .- x)
end

function compute_h(W, X)
    """
    Compute output: Take the inner product between our features 'X' and the weight
    matrix 'W'
    """
    return X * W
end

function predict(W, X)
    """
    Take the inner product between our features and weight matrix, 
    then pass this value through our sigmoid activation
    """
    preds = sigmoid_activation(compute_h(W, X))

    # apply a step function to threshold the outputs to binary
    # class labels
    preds[preds .<= 0.5] .= 0
    preds[preds .> 0] .= 1

    return preds
end

function compute_gradient(error, train_x)
    """
    This is the gradient descent update of "average negative loglikelihood" loss function. 
    In lab02 our loss function is "sum squared error".
    """

    gradient = transpose(train_x) * error
    
    return gradient
end

function train(W, train_x, train_y, learning_rate, num_epochs)
    losses = []
    for epoch in 1:num_epochs
        y_hat = sigmoid_activation(compute_h(W, train_x))
        error = y_hat - train_y
        append!(losses, mean(-1 .* train_y .* log.(y_hat) .- (1 .- train_y) .* log.(1 .- y_hat)))
        grad = compute_gradient(error, train_x) # grad is so big

        W -= learning_rate * grad

        if epoch == 1 || epoch % 50 == 0
            print("Epoch=$epoch; Loss=$(losses[end])\n")
        end
    end
    return W, losses
end






train_x, train_y = MNIST.traindata(Int64); 
test_x, test_y = MNIST.testdata(Int64);

train_x_flatten = Flux.flatten(train_x)'
test_x_flatten = Flux.flatten(test_x)'

size(train_x_flatten), size(train_y), size(test_x_flatten), size(test_y)


l_mean = compute_average_intensity(train_x_flatten, train_y);
bar(0:9, l_mean, legend=false)


# compute average intensity for each data sample
intensity = compute_average_intensity_each_sample(train_x_flatten, train_y)
size(intensity)

symmetry = compute_symmetry(train_x_flatten)
size(symmetry)

num_img = 10
img_flat = train_x_flatten[1:num_img,:]
img = [reshape(img_flat[i,:], (28,28))' for i in 1:num_img]
[colorview(Gray, Float32.(img[i])) for i in 1:num_img]

# create X_new by horizontal stack intensity and symmetry
train_x_new = hcat(intensity, symmetry)
size(train_x_new)




normalized_train_x, mean_, std_ = normalize(train_x_new)

s_mean = compute_average_intensity(normalized_train_x, train_y)
bar(0:9, s_mean, legend=false)

# change the label: y=1 -> stay unchanged, y!=1 -> y=0
train_y_new = reshape(deepcopy(train_y), (size(train_y)[1], 1))
train_y_new[train_y_new .!= 1] .= 0
size(train_y_new)

# contruct data by adding ones
add_one_train_x = hcat(ones(size(normalized_train_x)[1],), normalized_train_x)
size(add_one_train_x)


W = rand(Normal(), (size(add_one_train_x)[2], 1))

num_epochs=2000
learning_rate=0.00001

W, losses = train(W, add_one_train_x, train_y_new, learning_rate, num_epochs);

plot(1:num_epochs, losses, legend=false)


preds_train = predict(W, add_one_train_x)

train_y_new = reshape(train_y_new, length(train_y_new), 1)
acc = accuracy(preds_train, train_y_new)
p = precision(preds_train, train_y_new)
r = recall(preds_train, train_y_new)
f1 = 2*p*r/(p + r)

print(" acc: $acc\n precision: $p\n recall: $r\n f1_score: $f1\n")




# compute test_y_new
test_y_new = reshape(deepcopy(test_y), (size(test_y)[1], 1))
test_y_new[test_y_new .!= 1] .= 0


# compute test_intensity and test_symmetry to form test_x_new
test_intensity = compute_average_intensity_each_sample(test_x_flatten, test_y_new)
test_symmetry = compute_symmetry(test_x_flatten)
test_x_new = hcat(test_intensity, test_symmetry)



# normalize test_x_new to form normalized_test_x
normalized_test_x = normalize(test_x_new, mean_, std_)

# add column `ones` to test_x_new
add_one_test_x = hcat(ones(size(normalized_test_x)[1],), normalized_test_x) # ones(size(normalized_test_x)[1],)
size(add_one_test_x)

preds_test = predict(W, add_one_test_x)

test_y_new = reshape(test_y_new, length(test_y_new), 1)
acc = accuracy(preds_test, test_y_new)
p = precision(preds_test, test_y_new)
r = recall(preds_test, test_y_new)
f1 = 2*p*r/(p + r) 

print(" acc: $acc\n precision: $p\n recall: $r\n f1_score: $f1\n")