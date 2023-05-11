close all
clear all
folder = '.\Results\FC_result';
Fs = 1000;
% Plots figures for the time domain tests and also finds the avg MSE for
% whichever result you choose. 
%% find error array
mse_arr = [];
scale_arr = [];

for i=1:1000
    input  = open([folder '\input '  num2str(i) '.mat']);
    output = open([folder '\output ' num2str(i) '.mat']);
    file = input.file;
    input = input.x';
    input = input/max(input);
    output = output.x';
    output = output/max(output);
    [h_input,~] = freqz(input,1,Fs/2,Fs);
    [h_output,f] = freqz(output,1,Fs/2,Fs);
    h_input = h_input/max(abs(h_input));
    h_output = h_output/max(abs(h_output));
    
    [mse_arr(end+1),scale_arr(end+1)] = spectral_error(h_input(1:250),h_output(1:250));


end
mse_arr = mse_arr';
scale_arr = scale_arr';

%% find avg error
avg_error = sum(mse_arr)/1000;
max(scale_arr)


%% plot 5 rand

mkdir([folder '\img\'])
for i = 1:1
    j = ceil(rand*1000);
    input  = open(['.\' folder '\input '  num2str(j) '.mat']);
    output = open(['.\' folder '\output ' num2str(j) '.mat']);
    file = input.file;
    input = input.x';
    input = input/max(input);
    output = output.x';
    output = output/max(output);
    [h_input,~] = freqz(input,1,Fs/2,Fs);
    [h_output,f] = freqz(output,1,Fs/2,Fs);
    h_input = h_input/max(abs(h_input));
    h_output = h_output/max(abs(h_output));
    

    soundsc(input,1000)
    pause(1)
    soundsc(output,1000)
    
    
    
    figure
    fig = tiledlayout(3,1);
    title(fig, ['sample: ' file])
    nexttile;
    hold on
    
    graph_freq_response(h_input*scale_arr(j),f,12)
    title(['simulated vs model freq, MSE: ' num2str(mse_arr(j))])
    graph_freq_response(h_output,f,12)
    set(gca,'xminorgrid', 'on')
    ylim([-5 20])
    xlim([0,250])
    legend('Simulated','Model')
   
    hold off



    nexttile;
    
    plot(input);
    title('simulated listener RIR')
    xlabel('sample')
    ylabel('amplitude')
    nexttile;
    plot(output);
    title('model listener RIR')
    xlabel('sample')
    ylabel('amplitude')
    set(gcf, 'Position',  [100, 100, 1000, 1000])
    % saveas(gca,[folder '\img\fig_' num2str(i) '.png'])
    
end

