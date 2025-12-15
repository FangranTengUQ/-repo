function net = llrnetNeuralNetwork(hiddenLayerSize, varargin)
%llrnetNeuralNetwork Shallow neural network for LLR estimation
%   NET = llrnetNeuralNetwork(HLS) returns a function fitting
%   network, NET, which can be trained to estimate LLR values. HLS
%   is the size of the hidden layer.
%
%   NET = llrnetNeuralNetwork(HLS,USERELU) uses rectified linear unit (ReLU)
%   as the hidden layer activation function, if USERELU is true. If false,
%   the network uses sigmoid.
%
%   See also LLRNeuralNetworkExample, fitnet.

%   Copyright 2019 The MathWorks, Inc.

narginchk(1,2)

validateattributes(hiddenLayerSize, {'double','single'},...
  {'scalar','positive','integer'},...
  mfilename,...
  'HLS')

if nargin > 1
  useReLU = varargin{1};
else
  useReLU = false;
end

validateattributes(useReLU,{'logical'},{'scalar'},mfilename,'USERELU')

trainFcn = 'trainlm';  % Levenberg-Marquardt backpropagation
net = fitnet(hiddenLayerSize,trainFcn);

% Use rectified linear unit (ReLU) as activation (transfer) function.
% Default is sigmoid (tansig).
if useReLU
  net.layers{1}.transferFcn = 'tansig';
end
% net.layers{2}.transferFcn = 'tansig';
% Setup Division of Data for Training, Validation, Testing
net.divideFcn = 'dividerand';  % Divide data randomly
net.divideMode = 'sample';  % Divide up every sample
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;
net.inputs{1}.processFcns = {};  % 清除输入预处理函数
net.outputs{1}.processFcns = {}; % 清除输出后处理函数

% Choose a Performance Function
net.performFcn = 'mse';  % Mean Squared Error

% Do NOT show training progress. To show the training progress, set this
% value to true.
net.trainParam.showWindow = false;
