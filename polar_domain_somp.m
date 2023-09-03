function [x, support] = polar_domain_somp(A_bar, W, Y_bar, L_hat)
    
    % Compute whitened observation matrix
    gamma_w = A_bar*W;

    % Initialize residual vectors and support estimate
    residual = Y_bar; % Residual vector
    
    support = zeros(L_hat,1);  % Support vector
    x = 0; % Estimate vector
    
    for s=1:L_hat
        c = ctranspose(gamma_w)*residual; % Distributed correlation 
        temp = sum(abs(c).^2,2); % Find maximum projection support
        support(s) = find(temp == max(temp), 1); % Update common support
        
        x = pinv(gamma_w(:,support(1:s)))*Y_bar; % Project input signal onto subspace using current support
        residual = Y_bar - gamma_w(:,support(1:s))*x; % Update residual
    end
end