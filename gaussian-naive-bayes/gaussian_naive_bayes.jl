using MLDatasets, Plots
using Statistics, Distributions
using MLJ
using Random
using DataFrames

Random.seed!(2022)

function train_test_split(X, y, test_size=0.33)
    """
    Parameters:
    -----------
    X (Vector<array>) : features array
    y (Vector<string>) : lables array
    test_size (float) : proportion of test dataset
    """
    n = size(X)[1]
    idx = shuffle(1:n)
    train_size = 1 - test_size
    train_idx = view(idx, 1:floor(Int, train_size*n))
    test_idx = view(idx, (floor(Int, train_size*n)+1):n)
    return X[train_idx,:], y[train_idx], X[test_idx,:], y[test_idx]
end

#virtual function
function likelihood(_mean=nothing, _std=nothing ,data=nothing, hypo=nothing)
    # prototype
end

#update histogram for new data 
function update(_hist, _mean, _std, data)
    """
    P(hypo/data)=P(data/hypo)*P(hypo)*(1/P(data))
    """
    hist = copy(_hist)

    #Likelihood * Prior 

    for hypo in keys(hist)
        # we use gaussian distribution to estimate the likelihood
        hist[hypo] = hist[hypo] * likelihood(_mean, _std, data, hypo) # prior * likelihood
    end    

    #Normalization

    s = 0
    for hypo in keys(hist)
        s = s + hist[hypo] # calculate the normalization by sum all the likelihood (P(X) = P(X|Y1) + P(X|Y2) + P(X|Y3) + ....+ P(X|YN))
    end    
    
    for hypo in keys(hist)
        hist[hypo] = hist[hypo] / s # perform the normalize, finally we have the posterior probability
    end
    return hist
end
    

function maxHypo(hist)
    #find the hypothesis with maximum probability from hist

    best_hypo = 1
    for hypo in keys(hist)
        if hist[hypo] > hist[best_hypo] 
            best_hypo = hypo
        end
    end
    return best_hypo
end

function Gauss(std, mean, x)
    #Compute the Gaussian probability distribution function for x
    x = reshape(x, (1, 4))
    cal =  (1 ./ (std .* sqrt(2 * π))) .* exp.((-1/2 * ((x .- mean)./ std) .^2))  
    return cal

end


function likelihood(_mean=nothing, _std=nothing ,data=nothing, hypo=nothing)
    """
    Returns: res=P(data/hypo)
    -----------------
    Naive bayes:
        Atributes are assumed to be conditionally independent given the class value.
    """

    std=_std[hypo]
    mean=_mean[hypo]
    res=1
    data = [data]
    for i in size(data)[1] # The assumption of the independence of feature, each feature also have gaussian distribution
        out_prob_density_func = Gauss(std, mean, data[i])
        for prob_feat in out_prob_density_func
            res *= prob_feat
        end
    end
    return res     
end

function fit(X, y, _std=nothing, _mean=nothing, _hist=nothing)
    """Parameters:
    X: training data
    y: labels of training data
    """
    n=size(X,1)
    
    hist=Dict()
    mean=Dict()
    std=Dict()
    
    #separate  dataset into rows by class
    for hypo in Set(y)
        #rows have hypo label
        rows = X[y .== hypo, :]
        #histogram for each hypo

        probability = length(rows) / n # prior probability
        hist[hypo]=probability

        #Each hypothesis represented by its mean and standard derivation
        """mean and standard derivation should be calculated for each column (or each attribute)"""
        mean[hypo] = Statistics.mean(rows, dims=1) # each hypothesis has different means and standard deviation
        std[hypo] = Statistics.std(rows, dims=1)
        
    end
    println("shape of mean of each hypo: ", size(mean[0]))

    _mean=mean
    _std=std
    _hist=hist
    return _hist, _mean, _std
end

function plot_pdf(_hist)
    f = Plots.bar(_hist, kind="bar")
    display(f)
end

function _predict(_hist, _mean, _std, data, plot=true)
    """
    Predict label for only 1 data sample
    ------------
    Parameters:
    data: data sample
    plot: True: draw histogram after update new record
    -----------
    return: label of data
    """

    hist = update(_hist, _mean, _std, data) # the dimension of this consider single sample each time
    if (plot == true)
        plot_pdf(hist)
    end
    return maxHypo(hist) # find the hypothesis that have the highest probability for the prediction
end


function predict(_hist, _mean, _std, data)
    """Parameters:
    Data: test data
    ----------
    return labels of test data
    """
    pred=[]
    n_sample = size(data, 1)
    for i in 1:n_sample # go through each element
        push!(pred, _predict(_hist, _mean, _std, data[i,:]))   
    end
    
    return pred
end    

function multi_classes_conf_mat(gt, pred)
    num = size(gt)[1]
    mat = zeros((3, 3))
    # row is ground truth, col is pred 
    for i in 1:num
        mat[gt[i], pred[i]] += 1
    end 

    tp = zeros((3))
    fp = zeros((3))
    fn = zeros((3))

    for i in 1:3
        tp[i] = mat[i, i] 
        fp[i] = sum(mat[:, i]) - tp[i]
        fn[i] = sum(mat[i, :]) - tp[i]
    end
    
    return mat, tp, fp, fn
end

function recall(gt, pred)
    gt = deepcopy(gt)
    pred = deepcopy(pred)

    gt .+= 1
    pred .+= 1

    num = size(gt)[1]
    mat, tp, fp, fn = multi_classes_conf_mat(gt, pred)
    recall = mean(tp ./ (tp .+ fn)) # just average, no weighting
    return recall
end

function precision(gt, pred)
    gt = deepcopy(gt)
    pred = deepcopy(pred)

    gt .+= 1
    pred .+= 1

    num = size(gt)[1]
    mat, tp, fp, fn = multi_classes_conf_mat(gt, pred)
    precision = mean(tp ./ (tp .+ fp)) # just average, no weighting
    return precision
end    

function f1(gt, pred)
    rec = recall(gt, pred)
    pre = precision(gt, pred)
    f1 = 2 * (pre * rec) / (pre + rec) 
    return f1
end   


X = MLDatasets.Iris.features()
X = transpose(X)
y = MLDatasets.Iris.labels()
y = replace(y, "Iris-setosa" => 0, "Iris-versicolor" => 1, "Iris-virginica" => 2)

train_x, train_y, test_x, test_y = train_test_split(X, y, 0.33)

_hist, _mean, _std = fit(train_x, train_y)
plot_pdf(_hist)


#label of test_y[10]
print("Label of X_test[10]: ", test_y[10])

#update model and show histogram with X_test[10]:
print("\nOur histogram after update X_test[10]: ", _predict(_hist, _mean, _std, test_x[10,:]))


pred = predict(_hist, _mean, _std, test_x)
print("Accuracy of your Gaussian Naive Bayes model:", accuracy(test_y,pred))

println("Precision: ", precision(test_y,pred))
println("Recall: ", recall(test_y,pred))
println("F1: ", f1(test_y,pred))