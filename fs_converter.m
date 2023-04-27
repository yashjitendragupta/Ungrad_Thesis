function y = fs_converter(x, fs_old, fs_new)
    y = downsample(lowpass(x,fs_new/2,fs_old),fs_old/fs_new);
end