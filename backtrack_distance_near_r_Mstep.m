% Armijo backtracking line search for distance length
function [r_new, dictionary_new, dictionary_ft_new] = backtrack_distance_near_r_Mstep(theta_old, r_old, grad, dictionary_ft, w_mean_posterior, w_covariance_posterior, Y_bar, A_bar, N, par_len, kc, d)
    
    alpha = 1;
    tau = 0.5;
    c = 0.5;
    t = c*(norm(grad,2)^2);

    f_old = norm(Y_bar - dictionary_ft * w_mean_posterior, "fro")^2 + trace(dictionary_ft' * dictionary_ft * w_covariance_posterior);   

    while true
        r_new = ones(par_len,1)./(ones(par_len,1)./r_old - alpha*grad);

        dictionary_new = dictionary_creation_near(N, par_len, theta_old, r_new, kc, d);
        dictionary_ft_new = A_bar * dictionary_new;

        f_new = norm(Y_bar - dictionary_ft_new * w_mean_posterior, "fro")^2 + trace(dictionary_ft_new' * dictionary_ft_new * w_covariance_posterior);        
        if f_old - f_new >= alpha * t || alpha < 1e-16
            break;
        end 

        alpha = alpha * tau;
    end
end