close all
clear all
folder = '.\Results\freq_result';
%% find error arry
mse_arr = [];
scale_arr = [];

for i=1:1000
    input = open([folder '\input ' num2str(i) '.mat']);
    output = open([folder '\output ' num2str(i) '.mat']);
    file = input.file;
    input = input.h';
    input = input(1:250);
    output = output.h';
    output = output(1:250);
    
    [mse_arr(end+1),scale_arr(end+1)] = spectral_error(input,output);


end
%% find avg error
avg_error = sum(mse_arr)/1000;
max(scale_arr);

%% graph five rand

mkdir([folder '\img\'])
for i=1:5
    j = ceil(rand*1000);
    input = open([folder '\input ' num2str(j) '.mat']);
    output = open([folder '\output ' num2str(j) '.mat']);
    file = input.file;
    input = input.h';
    input = input(1:250);
    output = output.h';
    output = output(1:250);

    [h,f] = freqz(1,1,250,500);

    [error,scale] = spectral_error(input,output);   
    figure
 
    hold on
    graph_freq_response(input*scale,f,12)
    graph_freq_response(output,f,12)
    hold off
    title(['Simulated vs Model LFMR for ' file(1:length(file)-4) ', MSE: ' num2str(error)])
    ylim([-0 20])
    legend('Simulated','Model')
    set(gca,'xminorgrid', 'on')

    saveas(gca,[folder '\img\fig_' num2str(i) '.png'])
    


end
