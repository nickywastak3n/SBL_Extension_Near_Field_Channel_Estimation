function theta_new = angular_domain_sigw(N_iter, N, theta_init, A_bar, Y_bar, L_hat)
    theta_new = theta_init;
   
    for n=1:N_iter
        theta_old = theta_new;

        % Optimize angle vector
        F_tilda = dictionary_creation_far(N, L_hat, theta_old);
        
        Psi_tilda = A_bar*F_tilda;
        Psi_tilda_H = Psi_tilda';
        Psi_tilda_proj_inv = pinv(Psi_tilda_H*Psi_tilda);
        
        grad = zeros(L_hat, 1);
        for l=1:L_hat
            dW_dtheta = zeros(N, L_hat);
            dW_dtheta(:,l) = der_theta_signature_far(N, theta_old(l));            
            dPsi_dtheta = A_bar * dW_dtheta;
            
            dPsi_dtheta_proj_inv = -Psi_tilda_proj_inv * (dPsi_dtheta' * Psi_tilda + Psi_tilda_H * dPsi_dtheta) * Psi_tilda_proj_inv;
            dP_dtheta = dPsi_dtheta * Psi_tilda_proj_inv * Psi_tilda_H + Psi_tilda * dPsi_dtheta_proj_inv * Psi_tilda_H + Psi_tilda * Psi_tilda_proj_inv * dPsi_dtheta';
            
            grad(l) = real(-trace(Y_bar' * dP_dtheta * Y_bar));
        end
        
        alpha = backtrack_angle_far(theta_old, grad, Psi_tilda, Y_bar, A_bar, N, L_hat);
        theta_new = theta_old - alpha * grad;
    end
end