function simResults = llrnetQAMLLR(simParams)
%llrnetQAMLLR M-ary QAM LLR
%   R = llrnetQAMLLR(P) calculates exact LLR, approximate LLR, and neural
%   network LLR values for M-ary QAM modulation based on the simulation
%   parameters structure, P. P is a structure with following fields:
%
%   M               - Modulation order
%   SNRValues       - SNR values in dB as a scalar or row vector
%   HiddenLayerSize - Hidden layer size for neural network
%   UseReLU         - Flag to ese ReLU as activation in the hidden layer
%   NumSymbols      - Number of symbols to simulate
%
%   The output R is the results structure with following fields:
%
%   exactLLR        - Exact LLR
%   approxLLR       - Approximate LLR
%   predictedLLR    - LLRNet predicted LLR
%   RxSymbols       - Received symbols
%   M               - Modulation order
%   SNRValues       - SNR values in dB as a scalar or row vector
%   HiddenLayerSize - Hidden layer size for neural network
%   NumSymbols      - Number of symbols to simulate
%
%   See also llrnetPlotLLR, LLRNeuralNetworkExample.

%   Copyright 2019 The MathWorks, Inc.


if ~(isa(simParams,'struct') ...
    && isfield(simParams,'M') ...
    && isfield(simParams,'SNRValues') ...
    && isfield(simParams,'HiddenLayerSize') ...
    && isfield(simParams,'UseReLU') ...
    && isfield(simParams,'NumSymbols'))
  error('commdemos:llrnet:ExpectedSimParams', ...
    ['Expected P to be a structure with fields M, SNRValues, '...
    'HiddenLayerSize, UseReLU, and NumSymbols.'])
end

numSims = length(simParams);
for p=1:numSims
  M = simParams(p).M;
  SNRValues = simParams(p).SNRValues;
  hiddenLayerSize = simParams(p).HiddenLayerSize;
  useReLU = simParams(p).UseReLU;
  numSymbols = simParams.NumSymbols;
  
  simResults(p) = ...
    qamLLRPerformance(M, SNRValues, hiddenLayerSize, useReLU, numSymbols); %#ok<AGROW>
end
end

function simResults = ...
  qamLLRPerformance(M, SNRValues, hiddenLayerSize, useReLU, numSymbols)

k = log2(M);    % Bits per symbols
numSNRValues = length(SNRValues);
symOrder = llrnetQAMSymbolMapping(M);

const = qammod(0:M-1,M,symOrder,'UnitAveragePower',1);
maxConstReal = max(real(const));
maxConstImag = max(imag(const));

numBits = numSymbols*k;
exactLLR = zeros(numBits,numSNRValues);
rxSym = zeros(numSymbols,numSNRValues);
fprintf('Generating channel impaired %dQAM symbols and exact LLR values.\n', M)
parfor snrIdx = 1:numSNRValues
  SNR = SNRValues(snrIdx);
  noiseVariance = 10^(-SNR/10);
  sigma = sqrt(noiseVariance);
  
  maxReal = maxConstReal + 3*sigma;
  minReal = -maxReal;
  maxImag = maxConstImag + 3*sigma;
  minImag = -maxImag;
  
  r = (rand(numSymbols,1)*(maxReal-minReal)+minReal) + ...
    1i*(rand(numSymbols,1)*(maxImag-minImag)+minImag);
  rxSym(:,snrIdx) = r;
  
  exactLLR(:,snrIdx) = qamdemod(r,M,symOrder,...
    'UnitAveragePower',1,'OutputType','llr','NoiseVariance',noiseVariance);
end

nnInput = zeros(numSymbols,2,numSNRValues);
nnOutput = zeros(numSymbols,k,numSNRValues);
fprintf('Formatting %dQAM symbols as inputs and exact LLR values as outputs to the neural network for training.\n', M)
for snrIdx = 1:numSNRValues
  rxTemp = rxSym(:,snrIdx);
  rxTemp = [real(rxTemp) imag(rxTemp)];
  nnInput(:,:,snrIdx) = rxTemp;
  
  llrTemp = exactLLR(:,snrIdx);
  nnOutput(:,:,snrIdx) = reshape(llrTemp, k, numSymbols)';
end

trainedNetworks = cell(1,numSNRValues);
for snrIdx=1:numSNRValues
  x = nnInput(:,:,snrIdx)';
  y = nnOutput(:,:,snrIdx)';
  
  fprintf('Training for %d-QAM @ SNR = %1.1fdB\n', M, SNRValues(snrIdx))
  
  % Train the Network
  net = llrnetNeuralNetwork(hiddenLayerSize, useReLU);
  if parallelComputingLicenseExists()
    [net,tr] = train(net,x,y,'useParallel','yes');
  else
    [net,tr] = train(net,x,y);
  end
  
  % Store the trained network
  trainedNetworks{snrIdx} = net;
  
  fprintf('Training done for SNR = %1.1fdB with an MSE of %1.2e.\n',...
    SNRValues(snrIdx), tr.best_perf)
end

numBits = numSymbols*k;
d = randi([0 1], numBits, 1);

txSym = qammod(d,M,symOrder,'InputType','bit','UnitAveragePower',1);

fprintf('Simulate exact LLR, approximate LLR, and LLRNet values for %dQAM symbols.\n', M)
[exactLLR,approxLLR,predictedLLR] = deal(zeros(numBits,numSNRValues));
rxSym = zeros(length(txSym),numSNRValues);
parfor snrIdx = 1:numSNRValues
  SNR = SNRValues(snrIdx);
  sigmas = 10^(-SNR/10);
  r = awgn(txSym,SNR);
  rxSym(:,snrIdx) = r;
  
  exactLLR(:,snrIdx) = qamdemod(r,M,symOrder,...
    'UnitAveragePower',1,'OutputType','llr','NoiseVariance',sigmas);
  approxLLR(:,snrIdx) = qamdemod(r,M,symOrder,...
    'UnitAveragePower',1,'OutputType','approxllr','NoiseVariance',sigmas);
  
  net = trainedNetworks{snrIdx};
  x = [real(r) imag(r)]';
  tempLLR = net(x);
  predictedLLR(:,snrIdx) = reshape(tempLLR, numBits, 1);
end


simResults.exactLLR = exactLLR;
simResults.approxLLR = approxLLR;
simResults.predictedLLR = predictedLLR;
simResults.RxSymbols = rxSym;
simResults.M = M;
simResults.SNRValues = SNRValues;
simResults.HiddenLayerSize = hiddenLayerSize;
simResults.NumSymbols = numSymbols;
end
