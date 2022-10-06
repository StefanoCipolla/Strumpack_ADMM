clear all
clc
close all
addpath(genpath('../data'));
addpath(genpath('../build'));
% Dataset
inst = get_instances_svm();
% Test
for ii = 1:size(inst,1)
    disp(" ")
    disp("Solving "+inst(ii,1));
    filename = inst(ii,1);
    d=inst(ii,2);
    beta=inst(ii,3);
    exe_command = "cd ../build/; ./SVM_ADMM ../data/" + ...
              filename + " " + num2str(d) + " " + num2str(beta) + " " + "Gauss test" +...
              " > ../Matlab/Results/" + filename + ".txt";

    system(exe_command);
end

function inst = get_instances_svm()
inst=[
"2M", "2",      "100";
"susy_10Kn","8","100";    
"a8a","122",    "100" ; 
"w7a","300",    "100";   
"a9a","122",    "100"; 
"w8a","300",    "100"; 
%"ijcnn1","22",  "100"; 
%"cod_rna","8",  "100";  
%"skin_nonskin", "3", "1000";
%"rcv1_binary",  "47236", "100";
%"webspam_uni",  "254", "1000";
%"SUSY","18", "10000";
];


end
