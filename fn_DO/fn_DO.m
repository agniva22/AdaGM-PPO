%% Consensus
function dx= fn_DO(t,x)
if mod(t,10)<0.1
    display(t);
end

dx=zeros(length(x),1);

%% Graph Topology
n=5;
 
% A=[0 0.3 0 0 0;0 0 0.5 0 0;0 0 0 0.6 0;0 0 0 0 0.8;0.2 0 0 0 0]; % Directed-Unbalanced
A=[0 0 0 0 0.2;0.3 0 0 0 0;0 0.5 0 0 0;0 0 0.6 0 0;0 0 0 0.8 0]; % Directed-Unbalanced

% A=[0 1 0 0 0;0 0 1 0 0;0 0 0 1 0;0 0 0 0 1;1 0 0 0 0]; % Directed-Balanced
% A=[0 1 0 0 1;1 0 1 0 0;0 1 0 1 0;0 0 1 0 1;1 0 0 1 0]; % Undirected

%% Switching Graph %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Directed
% A1 = [0 0 0 0 1; 1 0 0 0 0; 0 0 0 0 0; 0 0 0 0 0; 0 0 0 0 0];
% A2 = [0 0 0 0 0; 0 0 0 0 0; 1 0 0 0 0; 0 0 1 0 0; 0 0 0 0 0];
% A3 = [0 0 0 0 0; 0 0 0 0 0; 0 0 0 0 0; 0 0 0 0 0; 0 0 0 0 0];
% A4 = [0 0 0 0 0; 0 0 0 0 0; 0 0 0 0 0; 0 0 0 0 0; 0 1 0 1 0];

% Undirected
A1 = [0 1 0 0 1; 1 0 0 0 0; 0 0 0 0 0; 0 0 0 0 0; 1 0 0 0 0];
A2 = [0 0 1 0 0; 0 0 0 0 0; 1 0 0 1 0; 0 0 1 0 0; 0 0 0 0 0];
A3 = [0 0 0 0 0; 0 0 0 0 0; 0 0 0 0 0; 0 0 0 0 0; 0 0 0 0 0];
A4 = [0 0 0 0 0; 0 0 0 0 1; 0 0 0 0 0; 0 0 0 0 1; 0 1 0 1 0];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% A = zeros(n,n); %B=zeros(n,2);
% if mod(t,8)<1.2
%     A = A1; %B = B1;
% elseif mod(t,8)>=1.2 && mod(t,8)<2
%     A = A2; %B = B2;
% elseif mod(t,8)>=2 && mod(t,8)<2.4
%     A = A3; %B = B3;
% elseif mod(t,8)>=2.4 && mod(t,8)<4
%     A = A1; %B = B1;
% elseif mod(t,8)>=4 && mod(t,8)<5.2
%     A = A4; %B = B4;
% elseif mod(t,8)>=5.2 && mod(t,8)<6
%     A = A2; %B = B2;
% elseif mod(t,8)>=6 && mod(t,8)<6.4
%     A = A3; %B = B3;
% else 
%     A = A4; %B = B4;
% end 

%% Laplacian

L=zeros(n,n);

for i=1:n
    for j=1:n
        if i==j
            L(i,j)=A(i,:)*ones(n,1);
        else
            L(i,j)=-A(i,j);
        end
    end
end

% w=n^2*[0.2516, 0.1509, 0.1258, 0.0943, 0.3774]';  % Left Eigenvector of L associated with 0 for Directed-Unbalanced
w=n^2*[0.3774, 0.2516, 0.1509, 0.1258, 0.0943]';  % Left Eigenvector of L associated with 0 for Directed-Unbalanced

% w=ones(n,1);  % Left Eigenvector of L associated with 0 for Directed-balanced

%% Dynamics

%% AM Consensus Cortes 2016 [x(1:n) and x(n+1:2*n)]
g1=2^(3/2)*1;

% Convex Functions
df_AM_2016=zeros(n,1);

% Use These
% df_AM_2016=[1,-2/(x(2,1)+2)^3,-3/(x(3,1)+3)^4,4*(x(4,1)+4)^3,0]'; % Power Law
% df_AM_2016=[2,-8/(x(2,1))^3,-9/(x(3,1))^4,4*(x(4,1))^3,0]'; % Power Law

df_AM_2016=[0, -9/(x(2,1))^4, -8/(x(3,1))^3, 2,4*(x(5,1))^3]'; % Power Law

% Algorithm
for i=1:n
    s1=0;s2=0;
    for j=1:n
        s1=s1+A(i,j)*(x(i,1)-x(j,1));
        s2=s2+A(i,j)*(x(n+i,1)-x(n+j,1));
    end
dx(i,1)= (-sign(A(i,:)*ones(n,1))*df_AM_2016(i,1)-w(i,1)*s2-w(i,1)*g1*s1)*1.8;
dx(n+i,1)= w(i,1)*s1*1.8;
end


%% GM Consensus inspired from Cortes 2016 [x(2*n+1:3*n) and x(3*n+1:4*n)]
g2=2^(3/2)*1;

% Convex Functions
df_GM_2016=zeros(n,1);

% Use These
% df_GM_2016=[1,-2/(x(2*n+2,1)+2)^3,-3/(x(2*n+3,1)+3)^4,4*(x(2*n+4,1)+4)^3,0]'; % Power Law
% df_GM_2016=[2,-8/(x(2*n+2,1))^3,-9/(x(2*n+3,1))^4,4*(x(2*n+4,1))^3,0]'; % Power Law

df_GM_2016=[0, -9/(x(2*n+2,1))^4, -8/(x(2*n+3,1))^3, 2,4*(x(2*n+5,1))^3]'; % Power Law

% Algorithm
for i=1:n
    s1=0;s2=0;
    for j=1:n
        s1=s1+A(i,j)*(log(x(2*n+i,1))-log(x(2*n+j,1)));
        s2=s2+A(i,j)*(log(x(3*n+i,1))-log(x(3*n+j,1)));
    end
dx(2*n+i,1)= (-x(2*n+i,1)^2*sign(A(i,:)*ones(n,1))*df_GM_2016(i,1)-w(i,1)*x(2*n+i,1)*s2-w(i,1)*g2*x(2*n+i,1)*s1)*1;
dx(3*n+i,1)= w(i,1)*x(3*n+i,1)*s1*1;
end



%% AM Consensus T-AC 2018 [x(4*n+1:5*n), x(5*n+1:6*n), and x(6*n+1:7*n)]

% Convex Functions
df_AM_2018=zeros(n,1);

% Use These
% df_AM_2018=[1,-2/(x(4*n+2,1)+2)^3,-3/(x(4*n+3,1)+3)^4,4*(x(4*n+4,1)+4)^3,0]'; % Power Law
% df_AM_2018=[2,-8/(x(4*n+2,1))^3,-9/(x(4*n+3,1))^4,4*(x(4*n+4,1))^3,0]'; % Power Law


df_AM_2018=[0,-9/(x(4*n+2,1))^4,-8/(x(4*n+3,1))^3,2,4*(x(4*n+5,1))^3]'; % Power Law

% Algorithm

beta=zeros(n,1);alpha=zeros(n,1);

alpha=x(6*n+1:7*n,1);
for i=1:n
    s1=0;s2=0;
    for j=1:n
        s1=s1+A(i,j)*(x(4*n+i,1)-x(4*n+j,1));
        s2=s2+A(i,j)*(x(5*n+i,1)-x(5*n+j,1));
    end

beta(i,1)=s1'*s1;

dx(4*n+i,1)= (-sign(A(i,:)*ones(n,1))*df_AM_2018(i,1)/w(i,1)-s2-(alpha(i,1)+beta(i,1))*s1)*0.58;
dx(5*n+i,1)= (alpha(i,1)+beta(i,1))*s1*0.58;
dx(6*n+i,1)= beta(i,1);
end



%% GM Consensus inspired from T-AC 2018 [x(7*n+1:8*n), x(8*n+1:9*n), and x(9*n+1:10*n)]

% Convex Functions
df_GM_2018=zeros(n,1);

% Use These
% df_GM_2018=[1,-2/(x(7*n+2,1)+2)^3,-3/(x(7*n+3,1)+3)^4,4*(x(7*n+4,1)+4)^3,0]'; % Power Law
% df_GM_2018=[2,-8/(x(7*n+2,1))^3,-9/(x(7*n+3))^4,4*(x(7*n+4,1))^3,0]'; % Power Law

df_GM_2018=[0, -9/(x(7*n+2,1))^4, -8/(x(7*n+3,1))^3, 2,4*(x(7*n+5,1))^3]'; % Power Law

beta_gm=zeros(n,1);alpha_gm=zeros(n,1);
alpha_gm=x(9*n+1:10*n,1);

% Algorithm
for i=1:n
    s1=0;s2=0;
    for j=1:n
        s1=s1+A(i,j)*(log(x(7*n+i,1))-log(x(7*n+j,1)));
        s2=s2+A(i,j)*(log(x(8*n+i,1))-log(x(8*n+j,1)));
    end
beta_gm(i,1)=w(i,1)*s1'*w(i,1)*s1;

dx(7*n+i,1)= (-x(7*n+i,1)^2*sign(A(i,:)*ones(n,1))*df_GM_2018(i,1)-x(7*n+i,1)*w(i,1)*s2-(alpha_gm(i,1)+beta_gm(i,1))*w(i,1)*x(7*n+i,1)*s1)*2;
dx(8*n+i,1)= w(i,1)*x(8*n+i,1)*s1*2;
dx(9*n+i,1)= beta_gm(i,1);
end

%% AM Consensus Kia 2015 Automatica [x(10*n+1:11*n) and x(11*n+1:12*n)]
g3=2^(3/2)*1;

% Convex Functions
df_AM_2015=zeros(n,1);

% Use These
% df_AM_2015=[1,-2/(x(2,1)+2)^3,-3/(x(3,1)+3)^4,4*(x(4,1)+4)^3,0]'; % Power Law
% df_AM_2015=[2,-8/(x(2,1))^3,-9/(x(3,1))^4,4*(x(4,1))^3,0]'; % Power Law

df_AM_2015=[0, -9/(x(10*n+2,1))^4, -8/(x(10*n+3,1))^3, 2,4*(x(10*n+5,1))^3]'; % Power Law

% Algorithm
for i=1:n
    s1=0;
    for j=1:n
        s1=s1+A(i,j)*(x(10*n+i,1)-x(10*n+j,1));
    end
dx(10*n+i,1)= (-sign(A(i,:)*ones(n,1))*df_AM_2015(i,1)-x(11*n+i,1)-w(i,1)*g3*s1)*2;
dx(11*n+i,1)= w(i,1)*s1*2;
end


%% GM Consensus Adaptive inspired from Kia 2015 Automatica [x(12*n+1:13*n) , x(13*n+1:14*n), and x(14*n+1:15*n),]
g3=2^(3/2)*1;

% Convex Functions
df_GM_2015=zeros(n,1);

% Use These
% df_AM_2015=[1,-2/(x(2,1)+2)^3,-3/(x(3,1)+3)^4,4*(x(4,1)+4)^3,0]'; % Power Law
% df_AM_2015=[2,-8/(x(2,1))^3,-9/(x(3,1))^4,4*(x(4,1))^3,0]'; % Power Law

df_GM_2015=[0, -9/(x(12*n+2,1))^4, -8/(x(12*n+3,1))^3, 2,4*(x(12*n+5,1))^3]'; % Power Law



beta_gm_1=zeros(n,1);alpha_gm=zeros(n,1);
alpha_gm_1=x(14*n+1:15*n,1);

% Algorithm
for i=1:n
    s1=0;
    for j=1:n
        s1=s1+A(i,j)*(log(x(12*n+i,1))-log(x(12*n+j,1)));
    end
beta_gm_1(i,1)=w(i,1)*s1'*w(i,1)*s1;


dx(12*n+i,1)= (-x(12*n+i,1)^2*sign(A(i,:)*ones(n,1))*df_GM_2015(i,1)-x(12*n+i,1)*log(x(13*n+i,1))-(alpha_gm_1(i,1)+beta_gm_1(i,1))*w(i,1)*x(12*n+i,1)*s1)*2;
dx(13*n+i,1)= w(i,1)*x(13*n+i,1)*s1*2;
dx(14*n+i,1)= beta_gm_1(i,1);

end
