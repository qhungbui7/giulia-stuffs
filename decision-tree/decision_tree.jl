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

function entropy(counts, n_samples)
    """
    Parameters:
    -----------
    counts: shape (n_classes): list number of samples in each class
    n_samples: number of data samples
    
    -----------
    return entropy 
    """
    sum = 0
    #sum of entropy of each class
    for i in counts
        if i != 0
            sum +=  - i / n_samples * log(i / n_samples)
        end
    end
    return sum
end

function entropy_of_one_division(division)
    """
    Returns entropy of a divided group of data
    Data may have multiple classes
    """

    n_samples = size(division, 1)
    n_classes = Set(division)
    
    counts = []
    #count samples in each class then store it to list counts
    for class in n_classes
        cnt = 0
        for sample in division
            if sample == class
                cnt += 1
            end
        end
        push!(counts, cnt)
    end
    

    entropy_val = entropy(counts, n_samples)
    return entropy_val, n_samples
end

function get_entropy(y_predict, y)
    """
    Returns entropy of a split
    y_predict is the split decision by cutoff, True/Fasle
    """
    n = size(y,1)
    entropy_true, n_true = entropy_of_one_division(y[y_predict]) # left hand side entropy
    entropy_false, n_false = entropy_of_one_division(y[.~y_predict]) # right hand side entropy
    # overall entropy
    #TODO: s=
    s = n_true / n * entropy_true + n_false / n * entropy_false
    return s
end


function fit(X, y, node=Dict(), depth=0)
    """
    Parameter:
    -----------------
    X: training data
    y: label of training data
    ------------------
    return: node 
    
    node: each node represented by cutoff value and column index, value and children.
        - cutoff value is thresold where you divide your attribute
        - column index is your data attribute index
        - value of node is mean value of label indexes, 
        if a node is leaf all data samples will have same label
    
    Note that: we divide each attribute into 2 part => each node will have 2 children: left, right.
    """
    
    #Stop conditions
    
    #if all value of y are the same 
    if all(y.==y[1])
        return Dict("val"=>y[1])
    else 
        col_idx, cutoff, entropy = find_best_split_of_all(X, y)    # find one split given an information gain 
        y_left = y[X[:,col_idx] .< cutoff]
        y_right = y[X[:,col_idx] .>= cutoff]
        node = Dict("index_col"=>col_idx,
                    "cutoff"=>cutoff,
                    "val"=>mean(y),
                    "left"=> Any,
                    "right"=> Any)
        left = fit(X[X[:,col_idx] .< cutoff, :], y_left, Dict(), depth+1)
        right= fit(X[X[:,col_idx] .>= cutoff, :], y_right, Dict(), depth+1)
        push!(node, "left" => left)
        push!(node, "right" => right)
        depth += 1 
    end
    return node
end

function find_best_split_of_all(X, y)
    col_idx = nothing
    min_entropy = 1
    cutoff = nothing

    for i in 1:size(X,2)
        col_data = X[:,i]
        entropy, cur_cutoff = find_best_split(col_data, y)
        if entropy == 0                   #best entropy
            return i, cur_cutoff, entropy
        elseif entropy <= min_entropy
            min_entropy = entropy
            col_idx = i
            cutoff = cur_cutoff
        end
    end
    return col_idx, cutoff, min_entropy
end

function find_best_split(col_data, y)
    """ 
    Parameters:
    -------------
    col_data: data samples in column
    """
    min_entropy = 10
    cutoff = 0
    #Loop through col_data find cutoff where entropy is minimum
    
    for value in Set(col_data)
        y_predict = col_data .< value
        my_entropy = get_entropy(y_predict, y)
        #min entropy=?, cutoff=?
        if min_entropy > my_entropy
            min_entropy = my_entropy 
            cutoff = value
        end

    end
    return min_entropy, cutoff
end


function predict(data, tree)
    pred = []
    n_sample = size(data, 1)
    for i in 1:n_sample
        push!(pred, _predict(data[i,:], tree))
    end
    return pred
end

function _predict(row, tree)
    cur_layer = tree
    while haskey(cur_layer, "cutoff")
            if row[cur_layer["index_col"]] < cur_layer["cutoff"]
                cur_layer = cur_layer["left"]
            else
                cur_layer = cur_layer["right"]
            end
        end
    if !haskey(cur_layer, "cutoff")
        return get(cur_layer, "val", false)
    end
end


X = MLDatasets.Iris.features()
X = transpose(X)
y = MLDatasets.Iris.labels()
y = replace(y, "Iris-setosa" => 0, "Iris-versicolor" => 1, "Iris-virginica" => 2)

train_x, train_y, test_x, test_y = train_test_split(X, y, 0.33)


tree = fit(train_x, train_y)
#label of test_y[10]
print("Label of X_test[10]: ", test_y[10])

#update model and show histogram with X_test[10]:
print("\nOur histogram after update X_test[10]: ", _predict(test_x[10,:], tree))

pred=predict(test_x, tree)
print("Accuracy of your Gaussian Naive Bayes model:", accuracy(test_y,pred))
