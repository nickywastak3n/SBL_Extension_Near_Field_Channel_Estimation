function [theta_new, r_new] = polar_domain_sigw(N_iter, N, theta_init, r_init, A_bar, Y_bar, L_hat, kc, d)
    
    theta_new = theta_init;
    r_new = r_init;

    for n=1:N_iter
        theta_old = theta_new;
        r_old = r_new;

        % Optimize angle vector
        W_tilda = dictionary_creation_near(N, L_hat, theta_old, r_old, kc, d);
        
        Psi_tilda = A_bar*W_tilda;
        Psi_tilda_H = Psi_tilda';
        Psi_tilda_proj_inv = pinv(Psi_tilda_H*Psi_tilda);
        
        grad = zeros(L_hat, 1);
        for l=1:L_hat
            dW_dtheta = zeros(N, L_hat);
            dW_dtheta(:,l) = der_theta_signature_near(N, theta_old(l), r_old(l), kc, d);            
            dPsi_dtheta = A_bar * dW_dtheta;
            
            dPsi_dtheta_proj_inv = -Psi_tilda_proj_inv * (dPsi_dtheta' * Psi_tilda + Psi_tilda_H * dPsi_dtheta) * Psi_tilda_proj_inv;
            dP_dtheta = dPsi_dtheta * Psi_tilda_proj_inv * Psi_tilda_H + Psi_tilda * dPsi_dtheta_proj_inv * Psi_tilda_H + Psi_tilda * Psi_tilda_proj_inv * dPsi_dtheta';
            
            grad(l) = real(-trace(Y_bar' * dP_dtheta * Y_bar));
        end
        
        alpha = backtrack_angle(theta_old, r_old, grad, Psi_tilda, Y_bar, A_bar, N, L_hat, kc, d);
        theta_new = theta_old - alpha * grad;
        
        % Optimize distance vector
        W_tilda = dictionary_creation_near(N, L_hat, theta_new, r_old, kc, d);

        Psi_tilda = A_bar*W_tilda;
        Psi_tilda_H = Psi_tilda';
        Psi_tilda_proj_inv = pinv(Psi_tilda_H*Psi_tilda);
        
        grad = zeros(L_hat, 1);
        for l=1:L_hat
            dW_dtheta = zeros(N, L_hat);
            dW_dtheta(:,l) = der_r_signature_near(N, theta_new(l), r_old(l), kc, d);            
            dPsi_dtheta = A_bar * dW_dtheta;
            
            dPsi_dtheta_proj_inv = -Psi_tilda_proj_inv * (dPsi_dtheta'*Psi_tilda + Psi_tilda_H*dPsi_dtheta) * Psi_tilda_proj_inv;
            dP_dtheta = dPsi_dtheta * Psi_tilda_proj_inv * Psi_tilda_H + Psi_tilda * dPsi_dtheta_proj_inv * Psi_tilda_H + Psi_tilda * Psi_tilda_proj_inv * dPsi_dtheta';
            
            grad(l) = real(-trace(Y_bar' * dP_dtheta * Y_bar));
        end

        alpha = backtrack_distance(theta_new, r_old, grad, Psi_tilda, Y_bar, A_bar, N, L_hat, kc, d);
        r_new = ones(L_hat,1)./(ones(L_hat,1)./r_old - alpha * grad);       
    end
end