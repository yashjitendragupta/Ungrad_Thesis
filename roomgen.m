%% params for generation
low_horiz = 3;
high_horiz = 12;
low_height = 2.5;
high_height = 5;
num_samples = 1000;
Fs = 1000;
% 48000 % Fs == 0 
% to downsample!
folder = 'high_res_test_set\';

%% vars for iterations
horiz_diff = high_horiz-low_horiz;
height_diff = high_height-low_height;

%% loop for generation

mkdir([folder 'time_integrated\'])
mkdir([folder 'time_listener\'])

for i=1:num_samples
    create_set(rand*horiz_diff + low_horiz,rand*horiz_diff + low_horiz, low_height+rand*height_diff, Fs,folder);
    clc
    fprintf([num2str(i) '/' num2str(num_samples) ' complete'])
end
fprintf('\ndone!')

samples_to_np





function create_set(xdim,ydim,zdim,Fs,folder)
    addpath('MCRoomSim-master\')

    %                       X
    %             X1        ^
    %        _______________|
    %       |               |           * Z axis increases upwards from origin
    %       |               |           * Z0 (floor)   
    %       |               |           * Z1 (ceiling)
    %    Y1 |               | Y0
    %       |               | 
    %       |               |
    %       |               |
    %    Y<-----------------0
    %             X0
    %

    %% Parameters
    % Dimensions for generated room
    % xdim = 6;
    % ydim = 9;
    % zdim = 3;
    dims = [xdim,ydim,zdim];
    % Fs = 44100;
    %% Room setup
    Absorption_freqs = 	[125, 250, 500, 1000, 2000, 4000];
    % plaster walls, suspended ceiling, carpet flooring
    Absorption_coeffs = [0.14,0.1,0.06,0.05,0.04,0.04;
                         0.14,0.1,0.06,0.05,0.04,0.04;
                         0.14,0.1,0.06,0.05,0.04,0.04;
                         0.14,0.1,0.06,0.05,0.04,0.04;
                         0.01,0.02,0.06,0.15,0.25,0.45;
                         0.15,0.11,0.04,0.04,0.07,0.08];
    Room = SetupRoom('Dim', dims,'Freq',Absorption_freqs,'Absorption', Absorption_coeffs);
    Options = MCRoomSimOptions('Fs',48000);
    
    %% Speaker placement
    % assume speaker is cardoid polar pattern, in the center of the wall in
    % front of the listener. Speaker is in the center of X1 wall, 25 cm from
    % the wall
    
    speaker_loc = [xdim-.25,ydim/2,zdim/2];
    
    % yaw pitch and roll, pointing towards the negative x direction.
    speaker_ypr = [180,0,0];
    
    % add speaker to room
    Sources = AddSource('Location', speaker_loc, 'Orientation', speaker_ypr, ...
        'Type', 'cardioid');
    
    %% Microphone placements
    % one microphone used as input to the machine learning algorithm will be
    % placed 25 cm below the speaker. The other is placed in the center of the
    % room, where the listener will theoretically be seated. This will be
    % placed a 1.2m from the ground.
    
    int_mic_loc = speaker_loc;
    int_mic_loc(3) = int_mic_loc(3) - .25;
    
    listener_mic_loc = [xdim/2,ydim/2,1.2];
    
    % Put mics in place
    
    Receivers = AddReceiver('Location', int_mic_loc);
    Receivers = AddReceiver(Receivers,'Location', listener_mic_loc);
    
    
    %% Run MCRoomSim
    
    [samples] = RunMCRoomSim(Sources,Receivers,Room,Options);
    
    %% Get data out of cells 
    
    integrated_IR = cell2mat(samples(1,1));
    listener_IR = cell2mat(samples(2,1));
    
    
    % soundsc(integrated_IR,Fs);
    % freqz(listener_IR);
    
    %% normalize data
    
    integrated_IR = integrated_IR./max(integrated_IR);
    listener_IR = listener_IR./max(listener_IR);

    integrated_IR = fs_converter(integrated_IR,48000,Fs);
    listener_IR = fs_converter(listener_IR,48000,Fs);
    %% Saving Dataset
    
    fname = [folder 'time_integrated\' num2str(xdim) 'x' num2str(ydim) 'x' num2str(zdim) '.wav'];
    audiowrite(fname, integrated_IR, Fs);
    fname = [folder 'time_listener\' num2str(xdim) 'x' num2str(ydim) 'x' num2str(zdim) '.wav'];
    audiowrite(fname, listener_IR, Fs);
end















