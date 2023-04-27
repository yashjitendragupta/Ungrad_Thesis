close all
clear all
addpath('MCRoomSim-master', "generatedirs");

%%acknowledgements
% using https://www.mathworks.com/matlabcentral/fileexchange/55161-1-n-octave-smoothing



%constants

Fs = 48000;
T = 1/Fs;



%absorbtion (wood floors with drywall and stucco ceiling
%https://www.acoustic-supplies.com/absorption-coefficient-chart/
floorabs = [0.04,0.04,0.07,0.06,0.06,0.07];
wallabs = [0.29,0.1,0.06,0.05,0.04,0.04];
ceilingabs = [0.14,0.1,0.06,0.05,0.04,0.04];
roomabs = [wallabs;wallabs;wallabs;wallabs;floorabs;ceilingabs];

% matrix of room dims, each row is a room, cols are x,y,z dims

x = 2 : 1.5 : 8
y = x
z = 3 : .25 : 4

sset = zeros(125, 3);

for i = 1:5
    for j = 1:5
        for k  = 1:5
            sset(25*(i-1)+(j-1)*5+(k-1)+1, :) = [x(i),y(j),z(k)]
        end
    end
end

% absorbtion for each room calculation

absorb = zeros(125,1);
for a = 1:125
    absorb(a) = absorb(a) + sset(a,1)*sset(a,2)*floorabs(3);
    absorb(a) = absorb(a) + sset(a,1)*sset(a,2)*ceilingabs(3);
    absorb(a) = absorb(a) + sset(a,1)*sset(a,3)*wallabs(3)*2; 
    absorb(a) = absorb(a) + sset(a,2)*sset(a,3)*wallabs(3)*2;
end

%t60 for each room

t60 = zeros(125,1);
for a = 1:125
    t60(a) = sset(a,1)*sset(a,2)*sset(a,3)*.16/absorb(a);
end

%mcroomsim test


for a = 1:125
    sources = AddSource('Location',[rand*sset(a,1),rand*sset(a,2),rand*sset(a,3)]);
    receivers = AddReceiver('Location',[rand*sset(a,1),rand*sset(a,2),rand*sset(a,3)]);
    room = SetupRoom('Dim',[sset(a,1),sset(a,2),sset(a,3)], ...
        'Absorption',roomabs);
    RIR = RunMCRoomSim(sources,receivers,room);
    tempname = sprintf('%gx%gx%g.wav', sset(a,1),sset(a,2),sset(a,3))
    tempname = strcat('generatedirs\',tempname);
    audiowrite(tempname,RIR,Fs);
end

%% room mode test



[y, Fs] = audioread('generatedirs/2x2x3.5.wav')

Y = fft(y);
L = length(y);
P2 = abs(Y/L);
P1 = P2(1:L/2+1);
P1(2:end-1) = 2*P1(2:end-1);
f = Fs*(0:(L/2))/L;
    
figure
    semilogx(f,P1) 
    title('Single-Sided Amplitude Spectrum of X(t)')
    xlabel('f (Hz)')
    ylabel('|P1(f)|')




%% function tests
close all


smoothed = octave_smoothing(P1,f,12);
smoothed_db = mag2db(smoothed);


figure
    semilogx(f,smoothed_db);
    title('1/12 smoothed IR')
    xlabel('frequency (hz)')
    ylabel('amplitude (dB)')

[modes,indices] = find_lf_modes(smoothed_db,f);


%% functions
function [lo, hi] = octave_smooth_window(freq, n)
    note = 12*log2(freq/55);
    lo = 55 * 2^((note-.5)/n);
    hi = 55 * 2^((note+.5)/n);
end

function output = octave_smoothing(IR,freqarr,n)
    freqstep = freqarr(2)-freqarr(1)
    output = zeros(length(IR),1);
    % Goes over the whole IR
    for i = 1:1:length(IR)
        % Finds window for the frequency we are at
        [lo,hi] = octave_smooth_window(freqarr(i), n);
        
        window = ceil((hi-lo)/freqstep);
        
        for j = 1:1:window
            if (i-j + floor(window/2)) <= length(IR)
                output(i) = output(i) + IR(i-j + floor(window/2))/window;
            end
        end 
    end
end

function [modes,indices] = find_lf_modes(arr,f)
    P1D = gradient(arr);
    lastval = 1;
    indices = [];
    modes = [];
    for i = 1:length(P1D)
        if (lastval * P1D(i) < 0)
            indices(length(indices)+1) = i;
        end
        lastval = P1D(i);
    end
    
    for i = 1:length(indices+1)
        modes(length(modes)+1) = f(i);
    end
end

