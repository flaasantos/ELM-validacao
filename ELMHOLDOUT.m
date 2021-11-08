clear all
clc

M=load('iris_log.dat');
M = M(randperm(size(M,1)),:);%embaralha as linhas

Treino=M(1:105,:);
Teste=M(106:end,:);      %matriz para teste(30%)

Treino_amostras = Treino(:,1:4)';
Treino_rotulos = Treino(:,5:7)';
Teste_amostras=Teste(:,1:4)';         %matriz de rótulos para Treino e teste
Teste_rotulos=Teste(:,5:7)';

%início do treino

Treino_amostras = [-ones(1,105);Treino_amostras]; %modifica a matriz de Treino com -1 na primeira linha
Teste_amostras = [-ones(1,45);Teste_amostras];%modifica a matriz de teste com -1 na primeira linha
q = 6; %numero de neurônios
p = 4; %numero de amostras

W = rand(q, p+1);  %matriz de valores aleatórios

Zaux = W*Treino_amostras;

Z = 1./(1+exp(Zaux));

Z = [-ones(1,105);Z];

M = Treino_rotulos*Z'*(Z*Z')^(-1);

%fim do treino

%início do teste

for i=1:45 %testa amostra por amostra da matriz de Testes e dá o resultado calculado e o original
X = Teste_amostras(:,i);
Rotulo_correto=Teste_rotulos(:,i)
z_1 = 1./(1+exp(W*X));

z_1 = [-1;z_1];

Rotulo_calculado = M*z_1

end

