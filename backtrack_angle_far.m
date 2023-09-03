% Armijo backtracking line search for angle length
function alpha = backtrack_angle_far(theta_old, grad, Psi_tilda_old, Y_bar, A_bar, N, L_hat)
    
    alpha = 1;
    tau = 0.5;
    c = 0.5;
    t = c * (norm(grad,2)^2);

    P_old = Psi_tilda_old * pinv(Psi_tilda_old);

    while true
        theta_new = theta_old - alpha*grad;
        F_tilda_new = dictionary_creation_far(N, L_hat, theta_new);

        Psi_tilda_new = A_bar * F_tilda_new;
        f_old = -trace(Y_bar' * P_old * Y_bar);
        f_new = -trace(Y_bar' * Psi_tilda_new * pinv(Psi_tilda_new) * Y_bar);

        if f_old - f_new >= alpha * t || alpha < 1e-16
            break;
        end 

        alpha = alpha * tau;
    end
end