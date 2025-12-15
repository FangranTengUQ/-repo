function dvb = getParamsDVBS2Demo(subsystemType, EsNodB, ...
    numLDPCDecIterations)
%getParamsDVBS2Demo DVB-S.2 link parameters
%   DVB = getParamsDVBS2Demo(TYPE, ESN0, NUMITER) returns DVB-S.2 link
%   parameters for subsystem type, TYPE, energy per symbol to noise power
%   spectral density ratio in dB, ESN0, and number of LDPC decoder
%   iterations, NUMITER. The output, DVB, is a structure with fields
%   specifying the parameter name and value.

% Copyright 2010-2018 The MathWorks, Inc.

validatestring(subsystemType, {'QPSK 1/4', ...
            'QPSK 1/3', 'QPSK 2/5', 'QPSK 1/2', 'QPSK 3/5', 'QPSK 2/3', ...
            'QPSK 3/4', 'QPSK 4/5', 'QPSK 5/6', 'QPSK 8/9', 'QPSK 9/10',...
            '8PSK 3/5', '8PSK 4/5', '8PSK 2/3', '8PSK 3/4', '8PSK 5/6', ...
            '8PSK 8/9', '8PSK 9/10', '16APSK 2/3', '16APSK 3/4', ...
            '16APSK 4/5', '16APSK 5/6', '16APSK 8/9', '16APSK 9/10', ...
            '32APSK 3/4', '32APSK 4/5', '32APSK 5/6', '32APSK 8/9', ...
            '32APSK 9/10'}, 'getParamsDVBS2Demo', 'TYPE', 1);
        
validateattributes(EsNodB, {'numeric'}, ...
    {'finite', 'scalar'}, 'getParamsDVBS2Demo', 'ESNO', 2);

validateattributes(numLDPCDecIterations, {'numeric'}, ...
    {'positive', 'integer', 'scalar'}, 'getParamsDVBS2Demo', 'NUMITER', 3);

systemInfo = regexp(subsystemType, ' ', 'split');
modulationType = char(systemInfo(1));
dvb.CodeRate = char(systemInfo(2));
codeRate = str2num(dvb.CodeRate); %#ok<ST2NM>

dvb.EsNodB = EsNodB;
dvb.ModulationType = modulationType;

%--------------------------------------------------------------------------
% Source

dvb.NumBytesPerPacket = 188;
byteSize = 8;
dvb.NumBitsPerPacket = dvb.NumBytesPerPacket * byteSize;

%--------------------------------------------------------------------------
% BCH coding

[dvb.BCHCodewordLength, ...
 dvb.BCHMessageLength, ...
 dvb.BCHGeneratorPoly] = getbchparameters(codeRate);
dvb.BCHPrimitivePoly = de2bi(65581, 'left-msb');
dvb.NumPacketsPerBBFrame =floor(dvb.BCHMessageLength/dvb.NumBitsPerPacket);
dvb.NumInfoBitsPerCodeword = dvb.NumPacketsPerBBFrame*dvb.NumBitsPerPacket;
dvb.BitPeriod = 1/dvb.NumInfoBitsPerCodeword;

%--------------------------------------------------------------------------
% LDPC coding

dvb.LDPCCodewordLength = 64800;
dvb.LDPCParityCheckMatrix = dvbs2ldpc(codeRate);
if isempty(numLDPCDecIterations)
    dvb.LDPCNumIterations = 50;
else
    dvb.LDPCNumIterations = numLDPCDecIterations;
end

%--------------------------------------------------------------------------
% Interleaver: Section 5.3.3, p. 23

% No interleaving (for BPSK and QPSK)
dvb.InterleaveOrder = (1:dvb.LDPCCodewordLength).';

switch modulationType
    case '8PSK'
        Ncol = 3;
        iTemp = reshape(dvb.InterleaveOrder, ...
            dvb.LDPCCodewordLength/Ncol, Ncol).';
        if codeRate == 3/5
            % Special Case - Figure 8
            iTemp = flipud(iTemp);
        end
        dvb.InterleaveOrder = iTemp(:);
    case '16APSK'
        Ncol = 4;
        iTemp = reshape(dvb.InterleaveOrder, ...
            dvb.LDPCCodewordLength/Ncol, Ncol).';
        dvb.InterleaveOrder = iTemp(:);
    case '32APSK'
        Ncol = 5;
        iTemp = reshape(dvb.InterleaveOrder, ...
            dvb.LDPCCodewordLength/Ncol, Ncol).';
        dvb.InterleaveOrder = iTemp(:);
end

%--------------------------------------------------------------------------
% Modulation

switch modulationType
    case 'BPSK'
        Ry = [+1; -1];
        dvb.Constellation = complex(Ry);
        dvb.SymbolMapping = [0 1];
        dvb.PhaseOffset = 0;
        warning(message('comm:getParamsDVBS2Demo:InvalidModulationType')); 
    case 'QPSK'
        Ry = [+1; +1; -1; -1];
        Iy = [+1; -1; +1; -1];
        dvb.Constellation = (Ry + 1i*Iy)/sqrt(2);
        dvb.SymbolMapping = [0 2 3 1];
        dvb.PhaseOffset = pi/4;
    case '8PSK'
        A = sqrt(1/2);
        Ry = [+A +1 -1 -A  0 +A -A  0].';
        Iy = [+A  0  0 -A  1 -A +A -1].';
        dvb.Constellation = Ry + 1i*Iy;
        dvb.SymbolMapping  = [1 0 4 6 2 3 7 5];
        dvb.PhaseOffset = 0;
    case '16APSK'
        dvb.Constellation = dvbsapskmod((0:15)', 16, 's2', ...
                        dvb.CodeRate, 'UnitAveragePower', true);
        dvb.SymbolMapping = [12 14 15 13 4 0 8 10 2  6 7 3 11 9 1 5];
        dvb.PhaseOffset = [pi/4 pi/12];
    case '32APSK'
        dvb.Constellation = dvbsapskmod((0:31)', 32, 's2', ...
                          dvb.CodeRate, 'UnitAveragePower', true);
        dvb.SymbolMapping = [15 13 29 31 14 6 7 5 4 12 28 20 21 23 22 30 ...
                              11 10 2 3 1 0 8 9 25 24 16 17 19 18 26 27];
        dvb.PhaseOffset = [pi/4 pi/12 pi/16];
    otherwise
        error(message('comm:getParamsDVBS2Demo:ModulationUnsupported'));
end

numModLevels = length(dvb.Constellation);
dvb.BitsPerSymbol = log2(numModLevels);
dvb.ModulationOrder = 2^dvb.BitsPerSymbol;
%--------------------------------------------------------------------------
% Complex scrambling sequence

dvb.SequenceIndex = 2;

%--------------------------------------------------------------------------
% Number of symbols per codeword

dvb.NumSymsPerCodeword = dvb.LDPCCodewordLength/dvb.BitsPerSymbol;

%--------------------------------------------------------------------------
% Noise variance for channel and estimate for LDPC coding

dvb.NoiseVar  = 1/(10^(dvb.EsNodB/10));
dvb.NoiseVarEst = dvb.NoiseVar/(2*sin(pi/numModLevels)); 
% Note that NoiseVarEst for QPSK is NoiseVar/(2*sqrt(2))

%--------------------------------------------------------------------------
% Delays

dvb.RecDelayPreBCH = dvb.BCHMessageLength;

%--------------------------------------------------------------------------
function [nBCH, kBCH, genBCH] = getbchparameters(R)

table5a = [1/4 16008 16200 12 64800 
           1/3 21408 21600 12 64800 
           2/5 25728 25920 12 64800
           1/2 32208 32400 12 64800
           3/5 38688 38880 12 64800
           2/3 43040 43200 10 64800
           3/4 48408 48600 12 64800
           4/5 51648 51840 12 64800
           5/6 53840 54000 10 64800
           8/9  57472 57600 8 64800
           9/10 58192 58320 8 64800];

rowidx = find(abs(table5a(:,1)-R)<.001);
kBCH = table5a(rowidx,2);
nBCH = table5a(rowidx,3);
tBCH = table5a(rowidx,4);

a8 = [1  0  0  0  1  1  1  0  0  0  0  0  0  0  1 ...
    1  1  0  0  1  0  0  1  0  1  0  1  0  1  1 ...
    1  1  1  0  1  1  1  0  0  0  1  0  0  1  0 ...
    0  1  1  1  1  0  0  1  0  1  1  1  1  0  1 ...
    1  1  1  0  1  0  0  0  1  1  0  0  1  1  1 ...
    1  1  1  1  0  0  0  1  1  0  1  1  0  1  0 ...
    1  1  1  0  1  0  1  0  0  0  0  0  1  0  0 ...
    1  1  1  1  1  0  0  1  0  1  1  0  0  1  1 ...
    0  0  0  1  0  1  0  1  1];

a10 = [1  0  1  1  0  0  0  0  0  0  0  0  1  0  1 ...
    0  1  0  0  0  0  1  1  0  0  1  1  1  0  1 ...
    1  0  1  1  1  1  1  1  1  0  0  0  0  1  0 ...
    1  0  1  0  0  0  1  1  0  0  1  1  0  0  0 ...
    1  1  1  1  1  0  1  1  0  1  0  1  0  0  1 ...
    1  1  1  0  0  0  0  1  0  1  0  1  1  1  0 ...
    0  0  0  0  0  1  1  1  1  1  0  1  1  1  1 ...
    1  1  0  1  0  0  0  1  0  0  1  0  0  0  1 ...
    1  0  0  0  0  0  0  0  1  1  0  1  1  1  0 ...
    0  0  1  0  1  1  1  0  1  1  0  1  1  0  0 ...
    1  0  1  1  0  0  1  0  0  0  1];

a12 = [1  0  1  0  0  1  1  1  0  0  0  1  0  0  1 ...
    1  0  0  0  0  0  1  1  1  0  1  0  0  0  0 ...
    0  1  1  1  0  0  0  0  1  0  0  0  1  0  1 ...
    1  1  0  0  0  1  0  1  0  0  0  1  0  0  0 ...
    1  1  1  0  0  0  1  0  1  0  0  0  0  1  1 ...
    0  0  1  1  1  1  0  0  1  0  1  1  0  0  1 ...
    1  0  1  1  0  0  0  1  1  0  1  1  1  0  0 ...
    0  0  1  1  0  1  0  1  0  0  0  0  1  0  0 ...
    0  1  0  0  0  1  0  0  1  0  0  0  0  0  0 ...
    1  1  0  1  0  0  0  1  1  1  1  0  0  0  0 ...
    1  0  1  1  1  1  1  0  1  1  1  0  1  1  0 ...
    0  1  1  0  0  0  0  0  0  0  1  0  0  1  0 ...
    1  0  1  0  1  1  1  1  0  0  1  1  1];

switch tBCH
case 8
    genBCH = a8;
case 10
    genBCH = a10;
case 12
    genBCH = a12;
end
