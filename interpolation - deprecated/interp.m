load handel.mat
y = y(1:numel(y)-1);
Y = fft(y);
z = zeros(numel(Y),1);
Y_padded_even = [Y(1:numel(Y)/2); zeros(numel(Y),1); Y(numel(Y)/2+1:numel(Y))];
Y_padded_odd = [Y; zeros(numel(Y),1)];
Y_pitched = Y_padded_even*2;
y_odd_interp = ifft(Y_padded_odd);
y_even_interp = ifft(Y_padded_even);
y_pitched = ifft(Y_pitched);
% soundsc(real(y_even_interp))
