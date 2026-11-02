%% DO 
clc;close all;clear all;

%% initial conditions of followers
n=5;                                          %no of agents


% initial states of agents
x_in=4 *[1,1.2,1.5,0.5,0.1]';
z_in=x_in;
% z_in=ones(n,1);



alpha_in=0.1*ones(n,1);

%% Consensus
IC=[x_in',z_in',x_in',z_in',x_in',z_in',alpha_in',x_in',z_in',alpha_in',x_in',z_in'*0, x_in',ones(1,n),alpha_in'];

%% ode45
t=100;
option=odeset('RelTol',1e-5,'AbsTol',1e-6);
[T,X]=ode45(@fn_DO,[0:0.01:t],IC,option);

%% Plots
%% AM Consensus Cortes 2016 [X(:,1:n) and X(:,n+1:2*n)]

figure(1)
plot(T,X(:,1),'b--')
hold on
plot(T,X(:,2),'r-.')
plot(T,X(:,3),'--','color', [0.5000, 0.3250, 0.0980])
plot(T,X(:,4),'-.','color', [0, 0.5, 0])
plot(T,X(:,5),'-','color', [0.4940, 0.1840, 0.5560])
set(gca,'FontSize',16)
set(ylabel('$$\mathbf{x_{i}}$$'),'Interpreter','latex','fontsize',20)
xlabel('Time ','fontsize',16)
grid on
set(legend('$$1$$','$$2$$','$$3$$','$$4$$','$$5$$'),'Interpreter','latex','fontsize',20)
axis([0 100 -1 9])

axes('Position',[.6 .52 .25 .25])
box on
plot(T,X(:,1),'b--')
hold on
plot(T,X(:,2),'r-.')
plot(T,X(:,3),'--','color', [0.5000, 0.3250, 0.0980])
plot(T,X(:,4),'-.','color', [0, 0.5, 0])
plot(T,X(:,5),'-','color', [0.4940, 0.1840, 0.5560])
set(gca,'FontSize',10)
grid on
axis([40 100 1.15 1.25])

% figure(2)
% plot(T,X(:,n+1),'b--')
% hold on
% plot(T,X(:,n+2),'r-.')
% plot(T,X(:,n+3),'--','color', [0.5000, 0.3250, 0.0980])
% plot(T,X(:,n+4),'-.','color', [0, 0.5, 0])
% plot(T,X(:,n+5),'-','color', [0.4940, 0.1840, 0.5560])
% set(gca,'FontSize',16)
% set(ylabel('$$\mathbf{z_{i}}$$'),'Interpreter','latex','fontsize',20)
% xlabel('Time ','fontsize',20)
% grid on
% set(legend('$$1$$','$$2$$','$$3$$','$$4$$','$$5$$'),'Interpreter','latex','fontsize',20)


%% GM Consensus ICARA 2026 (inspired from Cortes 2016) [X(:,2*n+1:3*n) and X(:,3*n+1:4*n)]

% figure(3)
% plot(T,X(:,2*n+1),'b--')
% hold on
% plot(T,X(:,2*n+2),'r-.')
% plot(T,X(:,2*n+3),'--','color', [0.5000, 0.3250, 0.0980])
% plot(T,X(:,2*n+4),'-.','color', [0, 0.5, 0])
% plot(T,X(:,2*n+5),'-','color', [0.4940, 0.1840, 0.5560])
% set(gca,'FontSize',16)
% set(ylabel('$$\mathbf{x_{i}}$$'),'Interpreter','latex','fontsize',20)
% xlabel('Time ','fontsize',16)
% grid on
% set(legend('$$1$$','$$2$$','$$3$$','$$4$$','$$5$$'),'Interpreter','latex','fontsize',20)
% 
% axes('Position',[.6 .5 .25 .25])
% box on
% plot(T,X(:,2*n+1),'b--')
% hold on
% plot(T,X(:,2*n+2),'r-.')
% plot(T,X(:,2*n+3),'--','color', [0.5000, 0.3250, 0.0980])
% plot(T,X(:,2*n+4),'-.','color', [0, 0.5, 0])
% plot(T,X(:,2*n+5),'-','color', [0.4940, 0.1840, 0.5560])
% set(gca,'FontSize',10)
% grid on
% axis([75 100 1.15 1.25])

% figure(4)
% plot(T,X(:,3*n+1),'b--')
% hold on
% plot(T,X(:,3*n+2),'r-.')
% plot(T,X(:,3*n+3),'--','color', [0.5000, 0.3250, 0.0980])
% plot(T,X(:,3*n+4),'-.','color', [0, 0.5, 0])
% plot(T,X(:,3*n+5),'-','color', [0.4940, 0.1840, 0.5560])
% set(gca,'FontSize',16)
% set(ylabel('$$\mathbf{z_{i}}$$'),'Interpreter','latex','fontsize',20)
% xlabel('Time ','fontsize',20)
% grid on
% set(legend('$$1$$','$$2$$','$$3$$','$$4$$','$$5$$'),'Interpreter','latex','fontsize',20)


%% AM Consensus T-AC 2018 [X(:,4*n+1:5*n), X(:,5*n+1:6*n), and X(:,6*n+1:7*n)]

figure(5)
plot(T,X(:,4*n+1),'b--')
hold on
plot(T,X(:,4*n+2),'r-.')
plot(T,X(:,4*n+3),'--','color', [0.5000, 0.3250, 0.0980])
plot(T,X(:,4*n+4),'-.','color', [0, 0.5, 0])
plot(T,X(:,4*n+5),'-','color', [0.4940, 0.1840, 0.5560])
set(gca,'FontSize',16)
set(ylabel('$$\mathbf{x_{i}}$$'),'Interpreter','latex','fontsize',20)
xlabel('Time ','fontsize',16)
grid on
set(legend('$$1$$','$$2$$','$$3$$','$$4$$','$$5$$'),'Interpreter','latex','fontsize',20)
axis([0 100 -6 15])

axes('Position',[.6 .52 .25 .25])
box on
plot(T,X(:,4*n+1),'b--')
hold on
plot(T,X(:,4*n+2),'r-.')
plot(T,X(:,4*n+3),'--','color', [0.5000, 0.3250, 0.0980])
plot(T,X(:,4*n+4),'-.','color', [0, 0.5, 0])
plot(T,X(:,4*n+5),'-','color', [0.4940, 0.1840, 0.5560])
set(gca,'FontSize',10)
grid on
axis([40 100 1.1 1.3])


% figure(6)
% plot(T,X(:,5*n+1),'b--')
% hold on
% plot(T,X(:,5*n+2),'r-.')
% plot(T,X(:,5*n+3),'--','color', [0.5000, 0.3250, 0.0980])
% plot(T,X(:,5*n+4),'-.','color', [0, 0.5, 0])
% plot(T,X(:,5*n+5),'-','color', [0.4940, 0.1840, 0.5560])
% set(gca,'FontSize',16)
% set(ylabel('$$\mathbf{z_{i}}$$'),'Interpreter','latex','fontsize',20)
% xlabel('Time ','fontsize',20)
% grid on
% set(legend('$$1$$','$$2$$','$$3$$','$$4$$','$$5$$'),'Interpreter','latex','fontsize',20)
% 
% figure(7)
% plot(T,X(:,6*n+1),'b--')
% hold on
% plot(T,X(:,6*n+2),'r-.')
% plot(T,X(:,6*n+3),'--','color', [0.5000, 0.3250, 0.0980])
% plot(T,X(:,6*n+4),'-.','color', [0, 0.5, 0])
% plot(T,X(:,6*n+5),'-','color', [0.4940, 0.1840, 0.5560])
% set(gca,'FontSize',16)
% set(ylabel('$$\alpha_{i}$$'),'Interpreter','latex','fontsize',20)
% xlabel('Time ','fontsize',20)
% grid on
% set(legend('$$1$$','$$2$$','$$3$$','$$4$$','$$5$$'),'Interpreter','latex','fontsize',20)

%% GM Consensus ICC2025 (inspired from T-AC 2018) [X(:,7*n+1:8*n), X(:,8*n+1:9*n), and X(:,9*n+1:10*n)]

figure(8)
plot(T,X(:,7*n+1),'b--')
hold on
plot(T,X(:,7*n+2),'r-.')
plot(T,X(:,7*n+3),'--','color', [0.5000, 0.3250, 0.0980])
plot(T,X(:,7*n+4),'-.','color', [0, 0.5, 0])
plot(T,X(:,7*n+5),'-','color', [0.4940, 0.1840, 0.5560])
set(gca,'FontSize',16)
set(ylabel('$$\mathbf{x_{i}}$$'),'Interpreter','latex','fontsize',20)
xlabel('Time ','fontsize',20)
grid on
set(legend('$$1$$','$$2$$','$$3$$','$$4$$','$$5$$'),'Interpreter','latex','fontsize',20)

axes('Position',[.6 .45 .25 .25])
box on
plot(T,X(:,7*n+1),'b--')
hold on
plot(T,X(:,7*n+2),'r-.')
plot(T,X(:,7*n+3),'--','color', [0.5000, 0.3250, 0.0980])
plot(T,X(:,7*n+4),'-.','color', [0, 0.5, 0])
plot(T,X(:,7*n+5),'-','color', [0.4940, 0.1840, 0.5560])
set(gca,'FontSize',10)
grid on
axis([40 100 1.18 1.22])

% figure(9)
% plot(T,X(:,8*n+1),'b--')
% hold on
% plot(T,X(:,8*n+2),'r-.')
% plot(T,X(:,8*n+3),'--','color', [0.5000, 0.3250, 0.0980])
% plot(T,X(:,8*n+4),'-.','color', [0, 0.5, 0])
% plot(T,X(:,8*n+5),'-','color', [0.4940, 0.1840, 0.5560])
% set(gca,'FontSize',16)
% set(ylabel('$$\mathbf{z_{i}}$$'),'Interpreter','latex','fontsize',20)
% xlabel('Time ','fontsize',20)
% grid on
% set(legend('$$1$$','$$2$$','$$3$$','$$4$$','$$5$$'),'Interpreter','latex','fontsize',20)
% 
% figure(10)
% plot(T,X(:,9*n+1),'b--')
% hold on
% plot(T,X(:,9*n+2),'r-.')
% plot(T,X(:,9*n+3),'--','color', [0.5000, 0.3250, 0.0980])
% plot(T,X(:,9*n+4),'-.','color', [0, 0.5, 0])
% plot(T,X(:,9*n+5),'-','color', [0.4940, 0.1840, 0.5560])
% set(gca,'FontSize',16)
% set(ylabel('$$\alpha_{i}$$'),'Interpreter','latex','fontsize',20)
% xlabel('Time ','fontsize',20)
% grid on
% set(legend('$$1$$','$$2$$','$$3$$','$$4$$','$$5$$'),'Interpreter','latex','fontsize',20)


%% AM Consensus Kia 2015 Automatica [X(:,10*n+1:11*n) and X(:,11*n+1:12*n)]

figure(11)
plot(T,X(:,10*n+1),'b--')
hold on
plot(T,X(:,10*n+2),'r-.')
plot(T,X(:,10*n+3),'--','color', [0.5000, 0.3250, 0.0980])
plot(T,X(:,10*n+4),'-.','color', [0, 0.5, 0])
plot(T,X(:,10*n+5),'-','color', [0.4940, 0.1840, 0.5560])
set(gca,'FontSize',16)
set(ylabel('$$\mathbf{x_{i}}$$'),'Interpreter','latex','fontsize',20)
xlabel('Time ','fontsize',20)
grid on
set(legend('$$1$$','$$2$$','$$3$$','$$4$$','$$5$$'),'Interpreter','latex','fontsize',20)

axes('Position',[.6 .45 .25 .25])
box on
plot(T,X(:,10*n+1),'b--')
hold on
plot(T,X(:,10*n+2),'r-.')
plot(T,X(:,10*n+3),'--','color', [0.5000, 0.3250, 0.0980])
plot(T,X(:,10*n+4),'-.','color', [0, 0.5, 0])
plot(T,X(:,10*n+5),'-','color', [0.4940, 0.1840, 0.5560])
set(gca,'FontSize',10)
grid on
axis([40 100 1.18 1.22])

% figure(12)
% plot(T,X(:,11*n+1),'b--')
% hold on
% plot(T,X(:,11*n+2),'r-.')
% plot(T,X(:,11*n+3),'--','color', [0.5000, 0.3250, 0.0980])
% plot(T,X(:,11*n+4),'-.','color', [0, 0.5, 0])
% plot(T,X(:,11*n+5),'-','color', [0.4940, 0.1840, 0.5560])
% set(gca,'FontSize',16)
% set(ylabel('$$\mathbf{x_{i}}$$'),'Interpreter','latex','fontsize',20)
% xlabel('Time ','fontsize',20)
% grid on
% set(legend('$$1$$','$$2$$','$$3$$','$$4$$','$$5$$'),'Interpreter','latex','fontsize',20)


%% GM Consensus inspired from Kia 2015 Automatica [X(:,12*n+1:13*n), X(:,13*n+1:14*n), and X(:,14*n+1:15*n)]

figure(13)
plot(T,X(:,12*n+1),'b--')
hold on
plot(T,X(:,12*n+2),'r-.')
plot(T,X(:,12*n+3),'--','color', [0.5000, 0.3250, 0.0980])
plot(T,X(:,12*n+4),'-.','color', [0, 0.5, 0])
plot(T,X(:,12*n+5),'-','color', [0.4940, 0.1840, 0.5560])
set(gca,'FontSize',16)
set(ylabel('$$\mathbf{x_{i}}$$'),'Interpreter','latex','fontsize',20)
xlabel('Time ','fontsize',20)
grid on
set(legend('$$1$$','$$2$$','$$3$$','$$4$$','$$5$$'),'Interpreter','latex','fontsize',20)

axes('Position',[.6 .45 .25 .25])
box on
plot(T,X(:,12*n+1),'b--')
hold on
plot(T,X(:,12*n+2),'r-.')
plot(T,X(:,12*n+3),'--','color', [0.5000, 0.3250, 0.0980])
plot(T,X(:,12*n+4),'-.','color', [0, 0.5, 0])
plot(T,X(:,12*n+5),'-','color', [0.4940, 0.1840, 0.5560])
set(gca,'FontSize',10)
grid on
axis([40 100 1.18 1.22])

% figure(14)
% plot(T,X(:,13*n+1),'b--')
% hold on
% plot(T,X(:,13*n+2),'r-.')
% plot(T,X(:,13*n+3),'--','color', [0.5000, 0.3250, 0.0980])
% plot(T,X(:,13*n+4),'-.','color', [0, 0.5, 0])
% plot(T,X(:,13*n+5),'-','color', [0.4940, 0.1840, 0.5560])
% set(gca,'FontSize',16)
% set(ylabel('$$\mathbf{x_{i}}$$'),'Interpreter','latex','fontsize',20)
% xlabel('Time ','fontsize',20)
% grid on
% set(legend('$$1$$','$$2$$','$$3$$','$$4$$','$$5$$'),'Interpreter','latex','fontsize',20)

%% Cost Plot

% f_AM_2016=zeros(length(T),1);
% f_GM_2016=zeros(length(T),1);
% f_AM_2018=zeros(length(T),1);
% f_GM_2018=zeros(length(T),1);
% for k=1:length(T)
%     % Use these
%     % Power Law
%     
% %     f_AM_2016(k,1)=[(2*X(k,1)), 4/(X(k,2))^2, 3/(X(k,3))^3, (X(k,4))^4, 5]*ones(n,1);
% %     f_GM_2016(k,1)=[(2*X(k,2*n+1)), 4/(X(k,2*n+2))^2, 3/(X(k,2*n+3))^3, (X(k,2*n+4))^4, 5]*ones(n,1);
% %     f_AM_2018(k,1)=[(2*X(k,4*n+1)), 4/(X(k,4*n+2))^2, 3/(X(k,4*n+3))^3, (X(k,4*n+4))^4, 5]*ones(n,1); 
% %     f_GM_2018(k,1)=[(2*X(k,7*n+1)), 4/(X(k,7*n+2))^2, 3/(X(k,7*n+3))^3, (X(k,7*n+4))^4, 5]*ones(n,1);
%     
%     
%     f_AM_2016(k,1)=[5, 3/(X(k,2))^3, 4/(X(k,3))^2, (2*X(k,4)), (X(k,5))^4]*ones(n,1);
%     
%     f_GM_2016(k,1)=[5, 3/(X(k,2*n+2))^3, 4/(X(k,2*n+3))^2, (2*X(k,2*n+4)), (X(k,2*n+5))^4]*ones(n,1);
%     f_AM_2018(k,1)=[5, 3/(X(k,4*n+2))^3, 4/(X(k,4*n+3))^2, (2*X(k,4*n+4)), (X(k,4*n+5))^4]*ones(n,1); 
%     f_GM_2018(k,1)=[5, 3/(X(k,7*n+2))^3, 4/(X(k,7*n+3))^2, (2*X(k,7*n+4)), (X(k,7*n+5))^4]*ones(n,1);
% 
% 
% end
% 
% Optimum=13.99*ones(1,length(T));

% figure(11)
% plot(T,f_GM_2018,'b--')
% hold on
% % plot(T,f_GM_2016,'r-.')
% hold on
% % plot(T,f_GM_2018,'r-.')
% plot(T,Optimum,'k--')
% set(gca,'FontSize',16)
% set(ylabel('$$\mathbf{f}$$'),'Interpreter','latex','fontsize',20)
% xlabel('Time ','fontsize',20)
% grid on
% axis([0 100 5 30]) 
% axes('Position',[.6 .6 .25 .25])
% box on
% plot(T,f_GM_2018,'b--')
% hold on
% plot(T,Optimum,'k--')
% set(gca,'FontSize',10)
% grid on
% axis([40 100 13.8 14.2]) 
% 
% figure(12)
% plot(T,f_AM_2016,'b--')
% hold on
% plot(T,Optimum,'k--')
% set(gca,'FontSize',16)
% set(ylabel('$$\mathbf{f}$$'),'Interpreter','latex','fontsize',20)
% xlabel('Time ','fontsize',20)
% grid on
% axis([0 100 5 35]) 
% axes('Position',[.6 .6 .25 .25])
% box on
% plot(T,f_AM_2016,'b--')
% hold on
% plot(T,Optimum,'k--')
% set(gca,'FontSize',10)
% grid on
% axis([40 100 13.6 14.3]) 
% 
% figure(13)
% plot(T,f_AM_2018,'b--')
% hold on
% plot(T,Optimum,'k--')
% set(gca,'FontSize',16)
% set(ylabel('$$\mathbf{f}$$'),'Interpreter','latex','fontsize',20)
% xlabel('Time ','fontsize',20)
% grid on
% axes('Position',[.6 .5 .25 .25])
% box on
% plot(T,f_AM_2018,'b--')
% hold on
% plot(T,Optimum,'k--')
% set(gca,'FontSize',10)
% grid on
% axis([40 100 13 16]) 
% 
% 
% % figure(12)
% % plot(T,abs(f_AM_2016-Optimum'),'b--')
% % hold on
% % plot(T,abs(f_GM_2016-Optimum'),'r-.')
% % % plot(T,abs(f_GM_2018-Optimum),'r-.')
% % set(gca,'FontSize',16)
% % set(ylabel('$$|\mathbf{f(x)}-\mathbf{f(x_*)}|$$'),'Interpreter','latex','fontsize',20)
% % xlabel('Time ','fontsize',20)
% % grid on
% 
% % figure(13)
% % plot(T,f_AM_2018,'-.','color', [0, 0.5, 0])
% % hold on
% % % plot(T,f_GM_2016,'r-.')
% % plot(T,f_GM_2018,'r-.')
% % plot(T,Optimum,'k--')
% % set(gca,'FontSize',16)
% % set(ylabel('$$\mathbf{f}$$'),'Interpreter','latex','fontsize',20)
% % xlabel('Time ','fontsize',20)
% % grid on
% 
% % figure(14)
% % plot(T,abs(f_AM_2018-Optimum'),'-.','color', [0, 0.5, 0])
% % hold on
% % % plot(T,abs(f_GM_2016-Optimum'),'r-.')
% % plot(T,abs(f_GM_2018-Optimum),'r-.')
% % set(gca,'FontSize',16)
% % set(ylabel('$$|\mathbf{f(x)}-\mathbf{f(x_*)}|$$'),'Interpreter','latex','fontsize',20)
% % xlabel('Time ','fontsize',20)
% % grid on


