%% Sample to NP

% Takes files from folder\time* and converts them to f domain w 250
% samples, one per hz, then saves them as mat arrays.
close all
clear all
addpath('mat2np')
% dataset name
folder = 'train_set/';
mkdir([folder 'freq_integrated/'])
mkdir([folder 'freq_listener/'])

% turns graphs on or off
graph     = true; 
only_five = true;

%% read and plot all the IR's from Dataset_1_integrated

fds = fileDatastore([folder 'time_integrated/*'], 'ReadFcn', @importdata);

files_integrated = fds.Files;
if(only_five)
    numFiles = 5;
else
    numFiles = length(files_integrated)
end
fds = fileDatastore([folder 'time_listener/*'], 'ReadFcn', @importdata);

files_listener = fds.Files;


% Loop over all files reading them in and plotting them.

for k = 1 : numFiles
    name = strsplit(files_integrated{k},'\');
    name = name{1,end};
    if(graph)
        figure
        fig = tiledlayout(2,1);
        title(fig, ['sample: ' name])

    end
    
%     Integrated 

    [y,Fs] = audioread(files_integrated{k});
    if(length(y) > 1000)
        y = y(1:1000);
    end
    [h,f] = freqz(y,1,Fs/2,Fs);

%     Normalize data
    h = abs(h); 
    h = h/max(abs(h));

%     save mat

    save([folder 'freq_integrated/' name(1:end-4) '.mat'],'h');

%     Graph
    if(graph)
        nexttile;
        graph_freq_response(h,f,12);
        title('integrated')
    end

%     listener
    [y,Fs] = audioread(files_listener{k});
    if(length(y) > 1000)
        y = y(1:1000);
    end
    [h,f] = freqz(y,1,Fs/2,Fs);
%     Normalize Data
    h = abs(h);
    h = h/max(h);

%     Save MAT
    save([folder 'freq_listener/' name(1:end-4) '.mat'], 'h');

%     Graph
    if(graph)
        nexttile;
        graph_freq_response(h,f,12);
        title('listener')
    end
end
