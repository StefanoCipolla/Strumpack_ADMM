
% Data From ijcnn12 
H = [91.4548, 92.3142, 92.3142; 88.6293, 86.649, 86.649; 13.2659, 90.3523, 90.3523]; 
xvalues = {'0.1','1','10'};
yvalues = {'0.1','1','10'};
h = heatmap(xvalues,yvalues,H,'FontSize',12);
h.Title = 'Dataset: ijcnn1';
h.XLabel = 'C';
h.YLabel = 'h';

set(gcf,'color','w');
filename = ['./ijcnn1_heatmap'];
keyboard
savefig(filename);
set(gcf, 'PaperPositionMode', 'auto');
print(filename,'-depsc2', '-r600'); 