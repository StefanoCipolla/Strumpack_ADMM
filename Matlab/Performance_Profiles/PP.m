clc; clear;
%% Loading Data


A_t      = [123.308;148.110;261.399;305.913;508.232;345.805;110.997;344.938;13354.384;Inf]; % LIBSVM
A_acc     =[83.953;97.904;93.247;82.697;99.444;96.007;90.374;99.960;99.081;Inf];

A_t_S      = ([135.923;2161.920;6319.780;256.032;10476.200;9.772;2.900;1127.79;5809.6;3938.68]+...
             [6.181;14.442;1.665;8.162;107.71;1.980;2.863;11.078;3.228;25.614])./3+...
             [0.300;0.486;0.173;0.471;1.498;0.470;0.444;1.219;0.909;9.471];
A_acc_S    = [83.314;97.465;89.940;83.477;97.679;92.403;89.305;99.846;95.551;72.338];



A_t_S2      = ([795.597;2311.330;14211.0;1176.99;10774.900;21.393;23.242;1232.730;7003.52;14495.9]+...
             [16.276;15.229;1.425;21.3909;124.076;2.041;2.377;7.560;5.640;159.972])./3+...
             [0.588;0.621;0.210;0.986;1.738;0.298;0.280;0.972;1.297;15.889];
A_acc_S2    = [83.476;97.465;87.921;83.643;97.672;92.314;89.308;99.855;96.123;72.047];


A_t_RACQP      = [98.269;82.838;67.830;206.527;348.122;427.551;531.787;4689.815;21669.329;Inf];
A_acc_RACQP    = [79.757;97.050;71.987;82.237;97.806;91.460;33.333;97.649;92.830;Inf];



num_of_probs = size(A_t_S,1);
c_t  = zeros(num_of_probs,4);
c_acc = zeros(num_of_probs,4);
%% Construction of Matrices for each metric

c_t(:,1) = A_t(1:num_of_probs,1);          %Libsvm
c_t(:,2) = A_t_S(1:num_of_probs,1);      %
c_t(:,3) = A_t_S2(1:num_of_probs,1);      %
c_t(:,4) = A_t_RACQP(1:num_of_probs,1);
c_t(isinf(c_t)) = NaN;

c_acc(:,1) = A_acc(1:num_of_probs,1);          %Libsvm
c_acc(:,2) = A_acc_S(1:num_of_probs,1);      %
c_acc(:,3) = A_acc_S2(1:num_of_probs,1);      %
c_acc(:,4) = A_acc_RACQP(1:num_of_probs,1);      %  
c_acc(isinf(c_acc)) = NaN;



%% Defining Colors

colors  = ['b' 'r' 'k' 'm' 'c' 'g','y'];   
lines   = {'-' ':' '--','-.'};
markers = [ 's' 'o' '^' 'v' 'p' '<' 'x' 'h' '+' 'd' '*' '>' ];

%% Time Performance Profile
[np,ns] = size(c_t);
minperf = min(c_t,[],2);
% Compute ratios and divide by smallest element in each row.
r = zeros(np,ns);
 for p = 1: np
   r(p,:) = c_t(p,:)/minperf(p);
 end
max_ratio = max(max(r));
% Replace all NaN's with twice the max_ratio and sort.
r(isnan(r)) = 2*max_ratio;
r = sort(r);
hl = zeros(ns,1);
% Plot stair graphs with markers.
subplot(1,2,1)
for s = 1: ns
[xs,ys] = stairs([1;r(:,s)],[0,(1:np)/np]);
  if (xs(1)==1)
       vv = find(xs==1,1,'last');
       xs = xs(vv:end);   ys = ys(vv:end);
   end
 sl = mod(s-1,4) + 1; 
 sc = mod(s-1,7) + 1; 
 sm = mod(s-1,12)+ 1;
 option1 = [char(lines(sl)) colors(sc) markers(sm)];
 hl(s) = semilogx(xs,ys,option1,'LineWidth', 4);
 hold on
end
axis([0 1.5*max_ratio 0 1]);
  twop = floor(log2(1.5*max_ratio));
  set(gca, ...
  'PlotBoxAspectRatio',[1 1 1]  , ...    
  'Box'         , 'on'          , ...
  'TickDir'     , 'in'          , ...
  'TickLength'  , [.02 .02]     , ...
  'YGrid'       , 'on'          , ...
  'TickDir'     , 'in'          , ...
  'TickLength'  , [0.02 0.02],...
  'LineWidth'   , 1            , ... 
  'FontSize'    , 22,...
  'XTick',2.^(0:twop))
title('\fontsize{20} Performance Profiles: Time')
h=legend('LIBSVM','HSS-ADMM (1)','HSS-ADMM (2)','RACQP',...
         'Position',[0.21767073450673 0.20649328742255 0.23339571393311 0.272679866914083]);
set(h,'Interpreter','latex','fontsize', 40)

% %% Accuracy PP
% [np,ns] = size(c_acc);
% minperf = min(1./c_acc,[],2);
% % Compute ratios and divide by smallest element in each row.
% r = zeros(np,ns);
%  for p = 1: np
%    r(p,:) = 1./c_acc(p,:)/minperf(p);
%  end
% max_ratio = max(max(r));
% % Replace all NaN's with twice the max_ratio and sort.
% r(isnan(r)) = 2*max_ratio;
% r = sort(r);
% hl = zeros(ns,1);
% % Plot stair graphs with markers.
% subplot(1,2,2)
% for s = 1: ns
% [xs,ys] = stairs([1;r(:,s)],[0,(1:np)/np]);
%   if (xs(1)==1)
%        vv = find(xs==1,1,'last');
%        xs = xs(vv:end);   ys = ys(vv:end);
%    end
%  sl = mod(s-1,4) + 1; 
%  sc = mod(s-1,7) + 1; 
%  sm = mod(s-1,12)+ 1;
%  option1 = [char(lines(sl)) colors(sc) markers(sm)];
%  hl(s) = semilogx(xs,ys,option1,'LineWidth', 4);
%  hold on
% end
% axis([0 1.5*max_ratio 0 1]);
%   twop = floor(log2(1.5*max_ratio));
%   set(gca, ...
%   'PlotBoxAspectRatio',[1 1 1]  , ...    
%   'Box'         , 'on'          , ...
%   'TickDir'     , 'in'          , ...
%   'TickLength'  , [.02 .02]     , ...
%   'YGrid'       , 'on'          , ...
%   'TickDir'     , 'in'          , ...
%   'TickLength'  , [0.02 0.02],...
%   'LineWidth'   , 1            , ... 
%   'FontSize'    , 22,...
%   'XTick',2.^(0:twop))
% title('\fontsize{20} Performance Profiles: (reciprocal of) Accuracy')
subplot(1,2,2)
H = [A_acc'; A_acc_S';A_acc_S2';A_acc_RACQP'];
H(isinf(H)) = 0;
xvalues = {'a8a','w7a','rcv1.binary','a9a','w8a','ijcnn1','cod.rna','skin.nonskin','webspam.uni','susy'};
yvalues = {'LIBSVM','HSS-ADMM (1)','HSS-ADMM (2)','RACQP'};
h = heatmap(xvalues,yvalues,H,'FontSize',12);
h.Title = '\fontsize{20} Accuracy';
%h.XLabel = 'Problem';
%h.YLabel = 'Solver';



set(gcf,'color','w');



%tightfig

pause
 filename = 'Summary';
     %savefig(filename);
     set(gcf, 'PaperPositionMode', 'auto');
     %set(gcf, 'Units', origfigunits);
     print(filename,'-depsc2');

close all;
% %% Objective Value Performance Profile
% [np,ns] = size(c_obj);
% minperf = min(c_obj,[],2);
% % Compute ratios and divide by smallest element in each row.
% r = zeros(np,ns);
%  for p = 1: np
%    r(p,:) = c_obj(p,:)/minperf(p);
%  end
% max_ratio = max(max(r));
% % Replace all NaN's with twice the max_ratio and sort.
% r(isnan(r)) = 2*max_ratio;
% r = sort(r);
% hl = zeros(ns,1);
% % Plot stair graphs with markers.
% subplot(2,2,3)
% for s = 1: ns
% [xs,ys] = stairs([1;r(:,s)],[0,(1:np)/np]);
%   if (xs(1)==1)
%        vv = find(xs==1,1,'last');
%        xs = xs(vv:end);   ys = ys(vv:end);
%    end
%  sl = mod(s-1,4) + 1; 
%  sc = mod(s-1,7) + 1; 
%  sm = mod(s-1,12)+ 1;
%  option1 = [char(lines(sl)) colors(sc) markers(sm)];
%  hl(s) = semilogx(xs,ys,option1,'LineWidth', 4);
%  hold on
% end
% axis([0 1.1*max_ratio 0 1]);
%   twop = floor(log2(1.1*max_ratio));
%   set(gca, ...
%   'Box'         , 'on'          , ...
%   'TickDir'     , 'in'          , ...
%   'TickLength'  , [0.02 0.02],...
%   'XTick',2.^(0:twop))
% title('\fontsize{18} Performance Profiles Based on Objective Value')




