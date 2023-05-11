% Given a freq. response and frequency array, graphs a decent looking
% semilog graph.

function graph_freq_response(H, f,oct)
    smoothed = smoothSpectrum(abs(H),f,oct);
    semilogx(f,mag2db(smoothed*10));
   
    grid on
    
    xticks([1 10 20 50 100 200 500])
    xticklabels({'1' '10' '20' '50' '100' '200' '500'})
    yticks([-40 -35 -30 -25 -20 -15 -10 -5 0 5 10 15])
    yticklabels({'-40','-35','-30','-25','-20','-15','-10','-5','0','5','10','15'})
    xlim([0 f(end)+1]);
    
    

    xlabel('Frequency')
    ylabel('magnitude (dB)')
    
end