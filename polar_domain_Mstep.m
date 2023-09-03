function [theta_new, r_new, dictionary_new, dictionary_ft_new] = polar_domain_Mstep(N, theta_old, r_old, dictionary_ft, Y_bar, A_bar, w_mean_posterior, w_covariance_posterior, par_len, kc, d)
    
    w_covariance_posterior = squeeze(sum(w_covariance_posterior, 1));    
    
    ddic_dtheta = zeros(N, par_len); % (N, H)
    for l=1:par_len
        ddic_dtheta(:, l) = der_theta_signature_near(N, theta_old(l), r_old(l), kc, d);            
    end

    ddic_ft_dtheta = A_bar * ddic_dtheta; % (PNrf, H)

    temp = ddic_ft_dtheta' * dictionary_ft; % (H, H)
    w_mean_posterior_proj = w_mean_posterior * w_mean_posterior'; % (H, H)
    
    grad = diag(-2*real(ddic_ft_dtheta' * Y_bar * w_mean_posterior'));
    grad = grad + diag(2*real(temp * w_mean_posterior_proj));
    grad = grad + diag(2*real(temp * w_covariance_posterior));      
   
    [theta_new, ~, dictionary_ft_new] = backtrack_angle_near_theta_Mstep(theta_old, r_old, grad, dictionary_ft, w_mean_posterior, w_covariance_posterior, Y_bar, A_bar, N, par_len, kc, d);
    
    ddic_dr = zeros(N, par_len);
    for l=1:par_len
        ddic_dr(:, l) = der_r_signature_near(N, theta_new(l), r_old(l), kc, d);            
    end

    ddic_ft_dr = A_bar * ddic_dr;

    temp = ddic_ft_dr' * dictionary_ft_new;
    w_mean_posterior_proj = w_mean_posterior * w_mean_posterior';
    
    grad = diag(-2*real(ddic_ft_dr' * Y_bar * w_mean_posterior'));
    grad = grad + diag(2*real(temp * w_mean_posterior_proj));
    grad = grad + diag(2*real(temp * w_covariance_posterior)); 

    [r_new, dictionary_new, dictionary_ft_new] = backtrack_distance_near_r_Mstep(theta_new, r_old, grad, dictionary_ft_new, w_mean_posterior, w_covariance_posterior, Y_bar, A_bar, N, par_len, kc, d);
end