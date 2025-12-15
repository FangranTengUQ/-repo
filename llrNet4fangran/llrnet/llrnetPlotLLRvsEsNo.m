function llrnetPlotLLRvsEsNo(perLLR,perApproxLLR,perLLRNet,...
    EsNoValues,subsystemType)
%llrnetPlotLLRvsEsNo Plot LLR vs EsNo

%   Copyright 2019 The MathWorks, Inc.

figure
semilogy(EsNoValues, perLLR(:,1), '-o')
hold on
semilogy(EsNoValues, perApproxLLR(:,1), '-*')
semilogy(EsNoValues, perLLRNet(:,1), '-+')
grid on
xlabel('E_s/N_o (dB)')
ylabel('PER')
title(sprintf('DVB-S.2 PER for %s', subsystemType))
legend('Exact LLR','Approx. LLR', 'LLRNet','Location','best')
