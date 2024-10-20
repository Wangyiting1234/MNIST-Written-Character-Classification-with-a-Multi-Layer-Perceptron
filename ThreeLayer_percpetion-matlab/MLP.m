
% Note: this file merely specifies the MLP class. It is not meant to be
% executed as a stand-alone script. The MLP needs to be instantiated and
% then used elsewhere, see e.g. 'testMLP131train.m'.

% A Multi-layer perceptron class
classdef MLP < handle
    % Member data
    properties (SetAccess=private)
        inputDimension % Number of inputs
        hiddenDimension % Number of hidden neurons
        outputDimension % Number of outputs
        hiddenLayerWeights % Weight matrix for the hidden layer, format (hiddenDim)x(inputDim+1) to include bias terms
        outputLayerWeights % Weight matrix for the output layer, format (outputDim)x(hiddenDim+1) to include bias terms
    end
    
    methods
        % Constructor: Initialize to given dimensions and set all weights
        % zero.
        % inputD ~ dimensionality of input vectors
        % hiddenD ~ number of neurons (dimensionality) in the hidden layer 
        % outputD ~ number of neurons (dimensionality) in the output layer 
        function mlp=MLP(inputD,hiddenD,outputD)
            mlp.inputDimension=inputD;
            mlp.hiddenDimension=hiddenD;
            mlp.outputDimension=outputD;
            mlp.hiddenLayerWeights=zeros(hiddenD,inputD+1);
            mlp.outputLayerWeights=zeros(outputD,hiddenD+1);
        end
        
        % TODO Implement a randomized initialization of the weight
        % matrices.
        % Use the 'stdDev' parameter to control the spread of initial
        % values.
        function mlp=initializeWeightsRandomly(mlp,stdDev)
            mlp.hiddenLayerWeights = normrnd(0,stdDev,[mlp.hiddenDimension,mlp.inputDimension+1]);% TODO
            mlp.outputLayerWeights=normrnd(0,stdDev,[mlp.outputDimension,mlp.hiddenDimension+1]);% TODO
        end
        
        % TODO Implement the forward-propagation of values algorithm in
        % this method.
        % 
        % inputData ~ a vector of data representing a single input to the
        % network in column format. It's dimension must fit the input
        % dimension specified in the contructor.
        % 
        % hidden ~ output of the hidden-layer neurons
        % output ~ output of the output-layer neurons
        % 
        % Note: the return value is automatically fit into a array
        % containing the above two elements
        function [hidden,hiddenAct,output,outputAct]=compute_forward_activation(mlp, inputData)
            % add new row in order to mulitply
            inputData = [inputData; 1];
            hidden = mlp.hiddenLayerWeights*inputData;
            hiddenAct = logsig(hidden);
            hiddenOut = [hiddenAct; 1];
            output = mlp.outputLayerWeights*hiddenOut;
            outputAct = logsig(output);
        end
        
        
        % This function calls the forward propagation and extracts only the
        % overall output. It does not have to be altered.
        function outputAct=compute_output(mlp,input)
            [hidden,hiddenAct,output,outputAct] = mlp.compute_forward_activation(input);
        end
        % This method implements MLP learning by means on backpropagation
        % of errors on a single data point.
        %
        
        % TODO Implement the backward-propagation of errors (learning) algorithm in
        % this method.
        function mlp = backward(mlp,input,labels,learningRate)
            [hidden,hiddenAct,output,outputAct] = mlp.compute_forward_activation(input);
            dOInput = logsig(output).*(1-logsig(output)).*(outputAct-labels);
            dOutputWeight = dOInput*[hiddenAct;1]';
            dHidden = mlp.outputLayerWeights'*dOInput;
            dHiddenInput = logsig(hidden).*(1-logsig(hidden)).*dHidden(1:length(hiddenAct));
            dHiddenWeight = dHiddenInput*[input;1]';
            mlp.updateWeight(dHiddenWeight,dOutputWeight,learningRate);
        end
        

        
        function mlp = modifyWeight(mlp,hiddenWeight, outputWeight)
            mlp.hiddenLayerWeights = hiddenWeight;
            mlp.outputLayerWeights = outputWeight;
        end 
        
        %   input (this is the supervision signal for learning)
        % learningRate ~ step width for gradient descent
        %
        % This method is expected to update mlp.hiddenLayerWeights and
        % mlp.outputLayerWeights.
        function mlp=updateWeight(mlp,dHiddenWeight,dOutputWeight, learningRate)
            mlp.outputLayerWeights = mlp.outputLayerWeights-(dOutputWeight*learningRate);
             mlp.hiddenLayerWeights =  mlp.hiddenLayerWeights-(dHiddenWeight*learningRate);
        end
        
        function mlp=train_single_data(mlp, inputData, targetOutputData, learningRate)
            mlp.backward(inputData,targetOutputData,learningRate);
     
        end
    end
end
