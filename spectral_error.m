function [output,scale]= spectral_error(h1,h2)
    h1abs = abs(h1);
    h2abs = abs(h2);
    error = [];
    for i = -1:.01:1
        error(end+1) = immse(h1abs*(100^i), h2abs);
    end
    [M,I] = min(error);
    output = M;
    scale = 100^((I-1)/100-1);
end