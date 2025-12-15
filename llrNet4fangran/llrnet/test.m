% 步骤1: 训练网络
X = [1 2 3; 4 5 6]';
Y = [1 0 1; 0 1 0]';
net = feedforwardnet(3);
net = train(net, X, Y);

% 步骤2: 手动计算
x_test = [0; 0;0];
W1 = net.IW{1}; b1 = net.b{1};
W2 = net.LW{2,1}; b2 = net.b{2};
x_processed = mapminmax('apply', x_test, net.inputs{1}.processSettings{1});
% 隐藏层计算
z1 = W1 * x_processed + b1;
a1 = tansig(z1);  % 或使用内置函数: a1 = feval(net.layers{1}.transferFcn, z1);

% 输出层计算
z2 = W2 * a1 + b2;
y_manual = purelin(z2);

% 步骤3: 对比结果
y_matlab = net(x_test);
disp(['手动计算: ', num2str(y_manual')]);
disp(['MATLAB输出: ', num2str(y_matlab')]);
disp(['最大误差: ', num2str(max(abs(y_manual - y_matlab)))]);