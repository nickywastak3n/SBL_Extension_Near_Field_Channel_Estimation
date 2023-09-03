% Channel Creation
function H = channel_creation(N, M, L, theta, r, gain, kc, km, d)
    H = zeros(N,M); % (N by M)

    for m=1:M
        for l=1:L
            H(:,m) = H(:,m) + sqrt(N/L)*gain(l)*exp(-1i*km(m)*r(l))*signature_near(N, theta(l), r(l), kc, d);
        end 
    end
end