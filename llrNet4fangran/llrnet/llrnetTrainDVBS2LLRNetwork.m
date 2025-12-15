function networks = llrnetTrainDVBS2LLRNetwork(...
  subsystemType, EsNoValues, numSymbols, hiddenLayerSize, varargin)
%llrnetTrainDVBS2LLRNetwork Train LLR neural network for DVB-S.2 system
%   NET = llrnetTrainDVBS2LLRNetwork(SUB,EsNo,N,HLS) trains neural
%   network(s) for each EsNo values and returns in NET. The neural network
%   is trained for the constellation defined by the subsystem type, SUB,
%   which must be one of
%
%        'QPSK 1/4', 'QPSK 1/3', 'QPSK 2/5', 'QPSK 1/2',
%        'QPSK 3/5', 'QPSK 2/3', 'QPSK 3/4', 'QPSK 4/5',
%        'QPSK 5/6', 'QPSK 8/9', 'QPSK 9/10'
%        '8PSK 3/5', '8PSK 4/5', '8PSK 2/3', '8PSK 3/4',
%        '8PSK 5/6', '8PSK 8/9', '8PSK 9/10'
%        '16APSK 2/3', '16APSK 3/4', '16APSK 4/5', '16APSK 5/6',
%        '16APSK 8/9', '16APSK 9/10'
%        '32APSK 3/4', '32APSK 4/5', '32APSK 5/6', '32APSK 8/9',
%        '32APSK 9/10'.
%
%   EsNo must be a scalar or a vector. N is the number of symbols used to
%   train the network. The neural networks has one hidden layer with HLS
%   neurons.
%
%   NET = llrnetTrainDVBS2LLRNetwork(SUB,EsNo,N,HLS,ReLU) uses a neural
%   network where the hidden layer activation function is a rectified
%   linear unit (ReLU) if ReLU is set to true. Otherwise, a sigmoid is used
%   as activation function.
%
%   See also LLRNeuralNetworkExample.

%   Copyright 2019-2022 The MathWorks, Inc.

narginchk(4,5)

validateattributes(EsNoValues,...
  {'double','single'},...
  {'row','finite','nonnan'},...
  mfilename,'EsNo')

validateattributes(numSymbols,...
  {'double','single'},...
  {'scalar','positive','integer'},...
  mfilename,'N')

validateattributes(hiddenLayerSize, {'double','single'},...
  {'scalar','positive','integer'},...
  mfilename,...
  'HLS')

if nargin > 4
  useReLU = varargin{1};
else
  useReLU = false;
end

validateattributes(useReLU,{'logical'},{'scalar'},mfilename,'USERELU')

maxNumLDPCIterations = 50;
numEsNoValues = length(EsNoValues);

networks = cell(1,numEsNoValues);
for EsNoIdx = 1:numEsNoValues
  EsNo = EsNoValues(EsNoIdx);
  fprintf('Training LLRNet for %s @ EsNo = %1.2f dB\n',subsystemType,EsNo)
  
  dvb = getParamsDVBS2Demo(subsystemType, EsNo, maxNumLDPCIterations);
  
  if dvb.ModulationOrder == 4 || dvb.ModulationOrder == 8
    const = dvb.Constellation;
  else
    const = dvbsapskmod((0:dvb.ModulationOrder-1)', dvb.ModulationOrder, 's2', ...
      dvb.CodeRate, 'UnitAveragePower', true);
  end
  
  maxConstReal = max(real(const));
  maxConstImag = max(imag(const));
  
  sigmas = 10^(-EsNo/10);
  maxReal = maxConstReal + 3*sqrt(sigmas);
  minReal = -maxReal;
  maxImag = maxConstImag + 3*sqrt(sigmas);
  minImag = -maxImag;
  
  xComplex = (rand(numSymbols,1)*(maxReal-minReal)+minReal) + ...
    1i*(rand(numSymbols,1)*(maxImag-minImag)+minImag);
  
  M = dvb.ModulationOrder;
  if M == 4 || M == 8
    y = pskdemod(xComplex,M,dvb.PhaseOffset,dvb.SymbolMapping, ...
      'OutputType','llr','NoiseVariance',dvb.NoiseVar);
  else
    y = dvbsapskdemod(xComplex, dvb.ModulationOrder, 's2', ...
      dvb.CodeRate, 'OutputType', 'llr', 'NoiseVar', ...
      dvb.NoiseVar, 'UnitAveragePower', true);
  end
  
  x = [real(xComplex) imag(xComplex)]';
  y = reshape(y, log2(dvb.ModulationOrder), []);
  
  net = llrnetNeuralNetwork(hiddenLayerSize, useReLU);
  
  perf = inf;
  netTemp = net;
  for p=1:3
    if parallelComputingLicenseExists()
      [netTemp,tr] = train(netTemp,x,y,'useParallel','yes');
    else
      [netTemp,tr] = train(netTemp,x,y);
    end
    fprintf('MSE for %s @ EsNo = %1.2f dB for iteration %d: %1.2e\n',...
      subsystemType,EsNo,p,tr.perf(end))
    if tr.perf(end) < perf
      perf = tr.perf(end);
      net = netTemp;
    end
    netTemp = init(netTemp);
  end
  fprintf('Final MSE for %s @ EsNo = %1.2f dB: %1.2e\n', ...
    subsystemType,EsNo,perf)
  networks{EsNoIdx} = net;
end