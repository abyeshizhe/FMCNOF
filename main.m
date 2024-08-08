clc
clear 
tic
%% ��������

%% ��������
viewcell=X;
Y = double(Y);
viewmat=[];
V=size(viewcell,2);
for i=1:V
    viewcell{i}=double(viewcell{i});
    viewcell{i}=mapstd(viewcell{i});
    viewcell0{i}= viewcell{i}; % չ������Ϊһ�У�Ȼ��ת��Ϊһ�С�
    viewcell{i} = reshape(viewcell0{i}, size(viewcell{i}));
    viewmat = [viewmat viewcell{i}];
end
s=length(unique(Y));  % sΪ������ĸ�
[n,~]=size(viewmat);
acc=[];nmi=[];purity=[];
m=s+3; %mnist4=s+3;cal7=s+7;cal20=s+7;calall=s+7;nus=s+9;nusw=s+9;awa=s+7;reuters=s+7
tic

for II=1:5
%% k-Means
[label, cluster_centers] = litekmeans(viewmat, m);
%% Get B
sigma=1000;
Bcell = GetB2(viewmat,cluster_centers,sigma,s,viewcell);
Bmat=cell2mat(Bcell);
for i=1:V
    Bcell{i}= Bcell{i}';
end
Bmat=cell2mat(Bcell);
%% ��ʼ��
Z=zeros(n,m);
B0=[];
alphaV1=[];
J=[];
alphaV=1/V*ones(1,V);%6
for i=1:V
    Z=Z+alphaV(i).*Bcell{i};
end

G=initializeG(m,s,1e-1);
F=Z*G;
for i=1:n
    F(i,:)=F(i,:)./sum(F(i,:));
end
for i=1:V
        Bmat=(Bcell{i})';
        Btemp=reshape(Bmat,m*n,1);
        B0=[B0 Btemp];
end
B=B0'*B0;
BB = max(B,B');
[vIM,dIM] = eig(BB);%�����AA��ȫ������ֵ�����ɶԽ���dIM������AA��������������vIM��������
if(min(min(dIM)) < 0)
    disp('Nonegative Definite Matrix;');
end

%% ��������
for iter=1:30  
    %�õ�G
    [U, S, W] = svd(F'*Z,'econ');
    G = (U*W')';
    %�õ�F
    F=Z*G;
    F(F<0)=0;
    %�õ�alpha 
    temp5=reshape(F*G',m*n,1);
    b=2*temp5'*B0;
    [alphaV, val,p] = SimplexQP_ALM(B, b', 1e-2,1.05,1);
    alphaV1=[alphaV1 alphaV];
    Z=zeros(n,m);
    for i=1:V
        Z=Z+alphaV(i)*Bcell{i};
    end
     J1=trace(Z'*Z)-2*trace(F'*Z*G)+trace(G*F'*F*G');
     J=[J J1];
end
[maxv,ind]=max(F,[],2);
[ACC,NMI,Purity] = ClusteringMeasure(Y, ind);
acc=[acc,ACC];
nmi=[nmi,NMI];
purity=[purity,Purity];
end
t=toc/5
[mean(acc),std(acc);mean(nmi),std(nmi);mean(purity),std(purity)]



