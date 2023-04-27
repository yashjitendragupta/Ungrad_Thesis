function graph_freq_response(H, f,oct)
    smoothed = smoothSpectrum(abs(H),f,oct);
    semilogx(f,mag2db(smoothed*10));
    xticks([1 10 100 500])
    xticklabels({'1','10','100','500'})
    grid on
    yticks([-40 -35 -30 -25 -20 -15 -10 -5 0 5 10 15])
    yticklabels({'-40','-35','-30','-25','-20','-15','-10','-5','0','5','10','15'})
    xlim([0 f(end)]);
    xlabel('Frequency')
    ylabel('magnitude (dB)')
    
end