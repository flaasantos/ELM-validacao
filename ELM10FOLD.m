clear all
clc

M=load('iris_log.dat');
M = M(randperm(size(M,1)),:);%embaralha as linhas


for k=0:14      %loop que modifica as matrizesde treino e teste
 %verifica a matriz de testes K vezes para encontrar os K-NN
    Teste_amostras=M(k*10+1:10*(k+1),1:4)';%matriz de testes, pega 10 amostras
    Teste_rotulos=M(k*10+1:10*(k+1),5:7)';
    aux=M;
    aux(k*10+1:10*(k+1),:)=[];
    Treino_amostras=aux(:,1:4)';    %o restante das amostras(140) ficam no treino
    Treino_rotulos=aux(:,5:7)';
    

%início do treino

Treino_amostras = [-ones(1,140);Treino_amostras]; %modifica a matriz de Treino com -1 na primeira linha
Teste_amostras = [-ones(1,10);Teste_amostras];%modifica a matriz de teste com -1 na primeira linha
q = 6; %numero de neurônios
p = 4; %numero de amostras

W = rand(q, p+1);  %matriz de valores aleatórios

Zaux = W*Treino_amostras;

Z = 1./(1+exp(Zaux));

Z = [-ones(1,140);Z];

MM = Treino_rotulos*Z'*(Z*Z')^(-1); %matriz de coeficientes

%fim do treino

%início do teste

for i=1:10 %testa amostra por amostra da matriz de Testes e dá o resultado calculado e o original
X = Teste_amostras(:,i);
Rotulo_correto=Teste_rotulos(:,i) %plota o rótulo dado
z_1 = 1./(1+exp(W*X));

z_1 = [-1;z_1];

Rotulo_calculado = MM*z_1 %plota o rótulo calculado

end
end
