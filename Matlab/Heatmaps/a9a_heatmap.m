
% Data From ijcnn12 
H = [76.3897, 76.8319, 76.8319; 82.5318, 83.4777, 83.4777; 82.2922, 81.6473, 81.6473]; 
xvalues = {'0.1','1','10'};
yvalues = {'0.1','1','10'};
h = heatmap(xvalues,yvalues,H,'FontSize',12);
h.Title = 'Dataset: a9a';
h.XLabel = 'C';
h.YLabel = 'h';

set(gcf,'color','w');
filename = ['./a9a_heatmap'];
keyboard
savefig(filename);
set(gcf, 'PaperPositionMode', 'auto');
print(filename,'-depsc2', '-r600'); 