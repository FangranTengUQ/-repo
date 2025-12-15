function [perLLR,perApproxLLR,perLLRNet] = ...
  llrnetDVBS2PER(subsystemType, EsNoValues, llrNets, numFrames, numErrors)
%llrnetDVBS2PER DVB-S.2 packet error rate
%   [PERLLR,PERAPRLLR,PERLLRNET] = llrnetDVBS2PER(SUB,EsNo,LLRNET,NF,NERR)
%   simulates the packet error rate (PER) of a DVB-S.2 system with
%   subsystem type, SUB, for EsNo values, EsNo. EsNo must be a scalar or a
%   vector. SUB must be one of
%
%        'QPSK 1/4', 'QPSK 1/3', 'QPSK 2/5', 'QPSK 1/2',
%        'QPSK 3/5', 'QPSK 2/3', 'QPSK 3/4', 'QPSK 4/5',
%        'QPSK 5/6', 'QPSK 8/9', 'QPSK 9/10'
%        '8PSK 3/5', '8PSK 4/5', '8PSK 2/3', '8PSK 3/4',
%        '8PSK 5/6', '8PSK 8/9', '8PSK 9/10'
%        '16APSK 2/3', '16APSK 3/4', '16APSK 4/5', '16APSK 5/6',
%        '16APSK 8/9', '16APSK 9/10'
%        '32APSK 3/4', '32APSK 4/5', '32APSK 5/6', '32APSK 8/9',
%        '32APSK 9/10'
%
%   The PER is calculated based on NF frames and NERR packet errors. LLRNet
%   contains the trained neural networks for LLR estimation. The outputs
%   PERLLR, PERAPPRLLR, and PERLLRNET are packet error rates for exact LLR,
%   approximate LLR, and LLRnet.
%
%   This function is based on the "DVB-S.2 Link, Including LDPC Coding"
%   example.
%
%   If a Parallel Computing Toolbox license is available, the simulations
%   are run in parallel using the available parallel pool.
%
%   See also LLRNeuralNetworkExample.

%   Copyright 2019-2022 The MathWorks, Inc.

narginchk(5,5)

validateattributes(numFrames,...
  {'double','single'},...
  {'scalar','positive','integer'},...
  mfilename,'NF')

validateattributes(numErrors,...
  {'double','single'},...
  {'scalar','positive','integer'},...
  mfilename,'NERR')

validateattributes(llrNets,...
  {'cell'}, ...
  {'row'}, ...
  mfilename,'LLRNET')

validateattributes(EsNoValues,...
  {'double','single'},...
  {'row','finite','nonnan'},...
  mfilename,'EsNo')

if ~isequal(size(llrNets), size(EsNoValues))
  error('commdemos:llrnet:EsNoLLRNetSizeMismatch', ...
    'EsNo and LLRNET must be same size')
end

for p=1:length(llrNets)
  assert(isa(llrNets{p},'network'),...
    'commdemos:llrnet:NetworkTypeMismatch', ...
    ['LLRNET must be a cell array of valid networks with the '...
    'same size as EsNo'])
end

numEsNoPoints = length(EsNoValues);
maxNumLDPCIterations = 50;
if parallelComputingLicenseExists()
  currentPool = gcp;
  if isempty(currentPool)
    currentPool = parpool;
  end
  numWorkers = currentPool.NumWorkers;
else
  numWorkers = 1;
end

[numPacketsLLR, numErrorsLLR, numPacketsApproxLLR,...
  numErrorsApproxLLR, numPacketsLLRNet, numErrorsLLRNet] ...
  = deal(zeros(numEsNoPoints, numWorkers));
parfor worker = 1:numWorkers
  t = datetime("now");
  [perEsNoLLR,perEsNoApproxLLR,perEsNoLLRNet] = deal(zeros(1,3));
  for esnoIdx = 1:numEsNoPoints
    EsNo = EsNoValues(esnoIdx); %#ok<PFBNS>
    dvb = getParamsDVBS2Demo(subsystemType, EsNo, maxNumLDPCIterations);

    M = dvb.ModulationOrder;
    
    % LDPC Encoder and Decoder Configuration
    ldpcenccfg = ldpcEncoderConfig(dvb.LDPCParityCheckMatrix);
    ldpcdeccfg = ldpcDecoderConfig(dvb.LDPCParityCheckMatrix);
    
    % BCH Encoder and Decoder
    encbch = comm.BCHEncoder('CodewordLength', dvb.BCHCodewordLength, ...
      'MessageLength', dvb.BCHMessageLength, ...
      'PrimitivePolynomialSource', 'Property', ...
      'PrimitivePolynomial', dvb.BCHPrimitivePoly, ...
      'GeneratorPolynomialSource', 'Property', ...
      'GeneratorPolynomial', dvb.BCHGeneratorPoly, ...
      'CheckGeneratorPolynomial', false);
    
    decbch = comm.BCHDecoder('CodewordLength', dvb.BCHCodewordLength, ...
      'MessageLength', dvb.BCHMessageLength, ...
      'PrimitivePolynomialSource', 'Property', ...
      'PrimitivePolynomial', dvb.BCHPrimitivePoly, ...
      'GeneratorPolynomialSource', 'Property', ...
      'GeneratorPolynomial', dvb.BCHGeneratorPoly, ...
      'CheckGeneratorPolynomial', false);
    
    net = llrNets{esnoIdx}; %#ok<PFBNS>
    
    % Additive White Gaussian Noise (AWGN) Channel
    chan = comm.AWGNChannel('NoiseMethod', 'Variance',...
      'Variance', dvb.NoiseVar);
    
    PERLLR = comm.ErrorRate;
    PERApproxLLR = comm.ErrorRate;
    PERLLRNet = comm.ErrorRate;
    
    bbFrameTx  = false(encbch.MessageLength,1);
    falseVec   = false(dvb.NumPacketsPerBBFrame, 1);
    frameCnt = 1;
    
    while (frameCnt <= numFrames/numWorkers) && ...
        (...
        (numErrorsLLR(esnoIdx,worker) < (numErrors/numWorkers)) || ...
        (numErrorsApproxLLR(esnoIdx,worker) < (numErrors/numWorkers)) || ...
        (numErrorsLLRNet(esnoIdx,worker) < (numErrors/numWorkers)) ...
        )
      
      % Random data bits
      bbFrameTx(1:dvb.NumInfoBitsPerCodeword) = ...
        logical(randi([0 1], dvb.NumInfoBitsPerCodeword, 1));
      
      % Encoding
      bchEncOut = encbch(bbFrameTx);
      ldpcEncOut = ldpcEncode(bchEncOut,ldpcenccfg);
      
      %Block Interleaver
      intrlvrOut = intrlv(ldpcEncOut, dvb.InterleaveOrder);
      
      % Modulation
      if M == 4 || M == 8
        modOut = pskmod(intrlvrOut,M,dvb.PhaseOffset,dvb.SymbolMapping, ...
          'InputType','bit');
      else
        modOut = dvbsapskmod(intrlvrOut, M, 's2', ...
          dvb.CodeRate, 'InputType', 'bit', 'UnitAveragePower', true);
      end
      
      % Channel
      chanOut = chan(modOut);
      
      % Prepare LLRNet input
      demodOutLLRNet = net([real(chanOut) imag(chanOut)]');
      demodOutLLRNet = demodOutLLRNet(:);
      
      % LLR calculations
      if M == 4 || M == 8
        demodOutLLR = pskdemod(chanOut,M,dvb.PhaseOffset,dvb.SymbolMapping, ...
          'OutputType','llr','NoiseVariance',dvb.NoiseVar);
        demodOutApproxLLR = pskdemod(chanOut,M,dvb.PhaseOffset,dvb.SymbolMapping, ...
          'OutputType','approxllr','NoiseVariance',dvb.NoiseVar);
      else
        demodOutLLR = dvbsapskdemod(chanOut, M, 's2', ...
          dvb.CodeRate, 'OutputType', 'llr', 'NoiseVar', ...
          dvb.NoiseVar, 'UnitAveragePower', true);
        demodOutApproxLLR = dvbsapskdemod(chanOut, M, 's2', ...
          dvb.CodeRate, 'OutputType', 'approxllr', 'NoiseVar', ...
          dvb.NoiseVar, 'UnitAveragePower', true);
      end
      
      % Deinterleave, decode, and check packet errors for exact LLR
      packetErr = packetError(demodOutLLR, bbFrameTx, dvb, ldpcdeccfg, decbch);
      perEsNoLLR = PERLLR(falseVec,   packetErr');
      numPacketsLLR(esnoIdx,worker) = perEsNoLLR(3);
      numErrorsLLR(esnoIdx,worker) = perEsNoLLR(2);
      
      % Deinterleave, decode, and check packet errors for approximate LLR
      packetErr = packetError(demodOutApproxLLR, bbFrameTx, dvb, ldpcdeccfg, decbch);
      perEsNoApproxLLR = PERApproxLLR(falseVec,   packetErr');
      numPacketsApproxLLR(esnoIdx,worker) = perEsNoApproxLLR(3);
      numErrorsApproxLLR(esnoIdx,worker) = perEsNoApproxLLR(2);
      
      % Deinterleave, decode, and check packet errors for LLRNet
      packetErr = packetError(demodOutLLRNet, bbFrameTx, dvb, ldpcdeccfg, decbch);
      perEsNoLLRNet = PERLLRNet(falseVec,   packetErr');
      numPacketsLLRNet(esnoIdx,worker) = perEsNoLLRNet(3);
      numErrorsLLRNet(esnoIdx,worker) = perEsNoLLRNet(2);
      
      frameCnt = frameCnt + 1;
    end
    fprintf('Worker #%d - Exact LLR -       Frame %d: EsNo: %1.2f, PER : %1.2e\n', ...
      worker, frameCnt, EsNo, perEsNoLLR(1))
    fprintf('Worker #%d - Approximate LLR - Frame %d: EsNo: %1.2f, PER : %1.2e\n', ...
      worker, frameCnt, EsNo, perEsNoApproxLLR(1))
    fprintf('Worker #%d - LLRNet          - Frame %d: EsNo: %1.2f, PER : %1.2e\n', ...
      worker, frameCnt, EsNo, perEsNoLLRNet(1))
    fprintf('Worker #%d - Elapsed time: %1.1fs\n',worker,seconds(datetime("now")-t))
  end
end
perLLR(:,1) = sum(numErrorsLLR, 2) ./ sum(numPacketsLLR, 2);
perLLR(:,2) = sum(numErrorsLLR, 2);
perLLR(:,3) = sum(numPacketsLLR, 2);

perApproxLLR(:,1) = sum(numErrorsApproxLLR, 2) ./ sum(numPacketsApproxLLR, 2);
perApproxLLR(:,2) = sum(numErrorsApproxLLR, 2);
perApproxLLR(:,3) = sum(numPacketsApproxLLR, 2);

perLLRNet(:,1) = sum(numErrorsLLRNet, 2) ./ sum(numPacketsLLRNet, 2);
perLLRNet(:,2) = sum(numErrorsLLRNet, 2);
perLLRNet(:,3) = sum(numPacketsLLRNet, 2);
end

function packetErr = packetError(demodOut, bbFrameTx, dvb, ldpcdeccfg, decbch)
%packetError Deinterleave, decode, and check packet errors
%   PE = packetError(Y,X,DVB,LDPCDEC,BCHDEC) deinterleaves, LDPC decodes,
%   BCH decodes received LLR values, Y, and compares to the transmitted
%   bits, X, and returns the packet error vector, PE.

deintrlvrOut = deintrlv(demodOut, dvb.InterleaveOrder);
ldpcDecOut = ldpcDecode(deintrlvrOut,ldpcdeccfg,dvb.LDPCNumIterations); % Default: Termination=early
bchDecOut = decbch(ldpcDecOut);
bbFrameRx = bchDecOut(1:dvb.NumInfoBitsPerCodeword,1);

% Error statistics
comparedBits = xor(bbFrameRx, bbFrameTx(1:dvb.NumInfoBitsPerCodeword));
packetErr    = any(reshape(comparedBits, dvb.NumBitsPerPacket, ...
  dvb.NumPacketsPerBBFrame));
end
