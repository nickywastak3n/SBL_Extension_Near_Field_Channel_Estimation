close all;
clear;
clc;

iteration = 25;
N_iter = 50;

%-------------------
% Parameter definition
N = 256; % # of BS antennas
Nrf = 4; % # of RF chains

c = 3e8; % Speed of light
fc = 100e9; % Carrier frequency
lambda = c/fc; % Wavelength
kc = 2*pi/lambda; % Center wave number

d = lambda/2; % Antenna spacing
B = 100e6; % Bandwidth
% K = 4; % # of Users
M = 4; % # of subcarriers
f_delta = B/M; % Subcarrier spacing
fm = (fc-B/2+f_delta/2)+f_delta*(0:M-1); % Subcarrier frequency
km = 2*pi*fm/c; % Subcarrier wave number

beta_delta = 1.2;
rho_min = 3; % Minimum allowable distance
L = 6; % # of paths
L_hat = 12;

P = 32; % Pilot vector length
SNR_vec = linspace(-5, 15, 10); % Signal to noise ratio vector

theta_dis = [-sqrt(3)/2, sqrt(3)/2]; % Uniform distribution of path angle
combiner_dis = [1/sqrt(N),-1/sqrt(N)];

% Result vector
NMSE_oracle_avg = zeros(length(SNR_vec), 1); % NMSE, Oracle LS result vector
NMSE_ls_avg = zeros(length(SNR_vec), 1); % NMSE, LS result vector
NMSE_asomp_avg = zeros(length(SNR_vec), 1); % NMSE, AOMP result vector
NMSE_psomp_avg = zeros(length(SNR_vec), 1); % NMSE, SOMP result vector
NMSE_asigw_avg = zeros(length(SNR_vec), 1); % NMSE, A-SIGW result vector
NMSE_psigw_avg = zeros(length(SNR_vec), 1); % NMSE, P-SIGW result vector
NMSE_sbl_near_avg = zeros(length(SNR_vec), 1); % NMSE, Polar SBL result vector

%-------------------
% Generate Angular Domain Transform matrix F
N_theta = 2*N; % # of sampled angles
theta_dic = (2*(0:N_theta-1)-N_theta+1)/N_theta;
F = dictionary_creation_far(N, N_theta, theta_dic);

%-------------------
% Generate Polar Domain Transform matrix W
Z_delta = (N*d/beta_delta)^2/2/lambda;
Z_rayleigh = 2*(N*d)^2/lambda;
S = fix(Z_delta/rho_min) + 1; % # of sampled distances

W = zeros(N,N_theta*S);
theta_dictionary = zeros(N_theta*S,1);
r_dictionary = zeros(N_theta*S,1);

for s=0:S-1
    theta_dic = (2*(0:N_theta-1)-N_theta+1)/N_theta;
    theta_dictionary(s*N_theta+1:(s+1)*N_theta) = theta_dic;
    
    if s==0
        r_dic = Z_rayleigh*ones(N_theta,1);        
        r_dictionary(s*N_theta+1:(s+1)*N_theta) = r_dic;
    else
        r_dic = Z_delta*(1-theta_dic.^2)/s;
        r_dictionary(s*N_theta+1:(s+1)*N_theta) = r_dic;
    end
    
    W(:,N_theta*s+1:N_theta*(s+1)) = dictionary_creation_near(N, N_theta, theta_dic, r_dic, kc, d);
end

% Parameters for SBL
alpha_th = 10; % Alpha threshold
a = 10^(-6);
b = 10^(-6);
C = 10^(-6);
D = 10^(-6);

%-------------------
% Start simulation

for itr = 1:iteration
    %-------------------
    % Random generation for each iteration...

    % Channel parameters
    theta_bs = unifrnd(theta_dis(1), theta_dis(2), 1, L);
    g_bs = (1/sqrt(2))*(randn(1,L) + 1i*randn(1,L));
    r_bs = unifrnd(5, 50, 1, L);

    % Analog combining matrix
    A = reshape(randsample(combiner_dis, P*Nrf*N, true), P*Nrf, N);

    % Whitening matrix
    Cw = []; % Covariance matrix
    for p=1:P
        Ap = A(Nrf*(p-1)+1:Nrf*p,:);
        Cw = blkdiag(Cw, Ap*Ap');
    end
    
    Dw = chol(Cw, "lower"); % Whitening matrix (P*Nrf by P*Nrf)

    %-------------------
    % For each SNR vector index
    for SNR_index=1:length(SNR_vec)
        SNR = SNR_vec(SNR_index); % SNR
        sigma = sqrt(10^(-SNR/10)); % Noise variance
        
        % Noise
        noise = zeros(P*Nrf,M);
        for m=1:M
            for p=1:P
                nmp = (sigma/sqrt(2))*(randn(N,1) + 1i*randn(N,1)); % (N by 1)
                Ap = A(Nrf*(p-1)+1:Nrf*p,:);
                noise(Nrf*(p-1)+1:Nrf*p,m) = Ap * nmp;
            end
        end
    
        % Output
        H = channel_creation(N, M, L, theta_bs, r_bs, g_bs, kc, km, d);
        normalize = norm(H, "fro")^2;
        Y = A*H + noise; % (P*Nrf by M)
        Y_bar = Dw\Y;
        A_bar = Dw\A;

        %-------------------
        % Oracle least square results
        oracle_dictionary = dictionary_creation_near(N, L, theta_bs, r_bs, kc, d);
        sparse_vec = pinv(A_bar * oracle_dictionary) * Y_bar;
        H_oracle = oracle_dictionary * sparse_vec;
        mse_oracle = norm(H-H_oracle,"fro")^2;
        NMSE_oracle_avg(SNR_index) = NMSE_oracle_avg(SNR_index) + mse_oracle/normalize;
    
        %-------------------
        % LS results
        H_ls = pinv(A_bar) * Y_bar;
        mse_ls = norm(H-H_ls,"fro")^2;
        NMSE_ls_avg(SNR_index) = NMSE_ls_avg(SNR_index) + mse_ls/normalize;

        %-------------------
        % A-SOMP results
        [x1, support1] = angular_domain_somp(A_bar, F, Y_bar, L_hat);
        H_asomp = F(:,support1)*x1;
        mse_asomp = norm(H-H_asomp, "fro")^2;
        NMSE_asomp_avg(SNR_index) = NMSE_asomp_avg(SNR_index) + mse_asomp/normalize;                

        %-------------------
        % P-SOMP results
        [x2, support2] = polar_domain_somp(A_bar, W, Y_bar, L_hat);
        
        H_psomp = W(:,support2)*x2;
        mse_psomp = norm(H-H_psomp, "fro")^2;
        NMSE_psomp_avg(SNR_index) = NMSE_psomp_avg(SNR_index) + mse_psomp/normalize;

        %-------------------
        % A-SIGW results
        theta_init_index = support1 - 1;
        theta_init = (2*theta_init_index-N_theta+1)/N_theta;        
        theta_refined = angular_domain_sigw(N_iter, N, theta_init, A_bar, Y_bar, L_hat);
        
        F_tilda = dictionary_creation_far(N, L_hat, theta_refined);
        Psi_tilda = A_bar*F_tilda;
        H_asigw = F_tilda * pinv(Psi_tilda) * Y_bar;
        mse_asigw = norm(H-H_asigw, "fro")^2;
        NMSE_asigw_avg(SNR_index) = NMSE_asigw_avg(SNR_index) + mse_asigw/normalize;

        %-------------------
        % P-SIGW results
        theta_init_index = rem(support2-1, N_theta);
        r_init_index = fix((support2-1)/N_theta);

        theta_init = (2*theta_init_index-N_theta+1)/N_theta;
        r_init = zeros(L_hat,1);
        for ind=1:L_hat
            if r_init_index(ind) == 0
                r_init(ind) = Z_rayleigh;
            else
                r_init(ind) = Z_delta*(1-theta_init(ind).^2)./r_init_index(ind);
            end
        end
        
        [theta_refined, r_refined] = polar_domain_sigw(N_iter, N, theta_init, r_init, A_bar, Y_bar, L_hat, kc, d);
        
        W_tilda = dictionary_creation_near(N, L_hat, theta_refined, r_refined, kc, d);
        Psi_tilda = A_bar*W_tilda;
        H_psigw = W_tilda * pinv(Psi_tilda) * Y_bar;
        mse_psigw = norm(H-H_psigw, "fro")^2;
        NMSE_psigw_avg(SNR_index) = NMSE_psigw_avg(SNR_index) + mse_psigw/normalize;

        %-------------------
        % Polar SBL results  
        N_temp = L_hat;
        theta_init = theta_dictionary(support2);
        r_init = r_dictionary(support2);
        dictionary_init = W(:, support2);
        Psi = A_bar * W;
        dictionary_ft_init = Psi(:, support2);

        alpha_mean_init = (a/b)*ones(N_temp, M); % (N_temp*S, M)
        beta_mean_init = C/D*ones(M, 1); % (M, 1)
        w_covariance_init = zeros(M, N_temp, N_temp); % (M, N_temp*S, N_temp*S)
        w_mean_init = zeros(N_temp, M); % (N_temp*S, M)
        for m=1:M
            w_covariance_init(m,:,:) = pinv(beta_mean_init(m) * (dictionary_ft_init' * dictionary_ft_init) + diag(alpha_mean_init(:, m)));
            w_mean_init(:,m) = beta_mean_init(m) * squeeze(w_covariance_init(m,:,:)) * dictionary_ft_init' * Y_bar(:,m); 
        end
        
        a_tilda = a + 1;
        c_tilda = C + P*Nrf;
                
        alpha_mean_posterior = alpha_mean_init;
        beta_mean_posterior = beta_mean_init;
        w_mean_posterior = w_mean_init;
        w_covariance_posterior = w_covariance_init;
        
        theta_posterior = theta_init;
        r_posterior = r_init;
        dictionary_posterior = dictionary_init;
        dictionary_ft_posterior = dictionary_ft_init;
        
        in = 1;
        while true
            %-------------------
            % E-step
            w_covariance_old = w_covariance_posterior; % (M, par_len, par_len)
            w_mean_old = w_mean_posterior; % (par_len, M)
            
            par_len = length(theta_posterior);
            dictionary = dictionary_posterior;
            dictionary_ft = dictionary_ft_posterior; % (P*Nrf, par_len)
            dictionary_ft_proj = dictionary_ft' * dictionary_ft;
            
            % Update alpha, beta according to w
            b_tilda_new = zeros(par_len, M); % (par_len, M)
            d_tilda_new = zeros(M, 1); % (M, 1)

            for m=1:M
                b_tilda_new(:,m) = diag(squeeze(w_covariance_old(m,:,:))) + abs(w_mean_old(:,m)).^2 + b;
                d_tilda_new(m) = norm(Y_bar(:,m)-dictionary_ft*w_mean_old(:,m), "fro").^2 + trace(dictionary_ft_proj*squeeze(w_covariance_old(m,:,:))) + D;         
            end
            
            beta_mean_new = real(c_tilda ./ d_tilda_new); % (M, 1)
            alpha_mean_new = real(a_tilda ./ b_tilda_new); % (par_len, M)
            
            % Update w according to alpha, beta
            w_covariance_new = zeros(M, par_len, par_len);
            w_mean_new = zeros(par_len, M);
            for m=1:M
                w_covariance_new(m,:,:) = inv(beta_mean_new(m) * dictionary_ft_proj + diag(alpha_mean_new(:, m)));
                w_mean_new(:,m) = beta_mean_new(m) * squeeze(w_covariance_new(m,:,:)) * dictionary_ft' * Y_bar(:,m); 
            end          
            
            % Update alpha, beta, w posterior vectors outside loop
            alpha_mean_posterior = alpha_mean_new;
            beta_mean_posterior = beta_mean_new;
            w_mean_posterior = w_mean_new;
            w_covariance_posterior = w_covariance_new;
            
            %-------------------
            % Pruning step
            pruned_index = max(alpha_mean_posterior, [], 2) <= alpha_th;
            
            theta_posterior = theta_posterior(pruned_index, :);
            r_posterior = r_posterior(pruned_index, :);
            par_len = length(theta_posterior);
            dictionary_ft = dictionary_ft(:, pruned_index);

            w_mean_posterior = w_mean_posterior(pruned_index, :);
            w_covariance_posterior = w_covariance_posterior(:, pruned_index, pruned_index);
            
            %-------------------
            % M-step
            [theta_posterior, r_posterior, dictionary_posterior, dictionary_ft_posterior] = polar_domain_Mstep(N, theta_posterior, r_posterior, dictionary_ft, Y_bar, A_bar, w_mean_posterior, w_covariance_posterior, par_len, kc, d);            
            
            in = in + 1;
            if in > N_iter
                break;
            end
        end

        % disp(strcat("Number of pruned columns: ", string(length(theta_posterior))))

        H_sbl_near = dictionary_posterior * w_mean_posterior;
        mse_sbl_near = norm(H-H_sbl_near, "fro")^2;
        NMSE_sbl_near_avg(SNR_index) = NMSE_sbl_near_avg(SNR_index) + mse_sbl_near/normalize;
       
        % Print in Command window...    
        disp(strcat("Iteration complete. (#: ", string(itr), ", SNR: ", string(SNR_vec(SNR_index)), " dB)"))      
    end
end

%-------------------
% Process and plot result
NMSE_oracle_avg = 10*log10(NMSE_oracle_avg/iteration);
NMSE_ls_avg = 10*log10(NMSE_ls_avg/iteration);
NMSE_asomp_avg = 10*log10(NMSE_asomp_avg/iteration);
NMSE_psomp_avg = 10*log10(NMSE_psomp_avg/iteration);
NMSE_asigw_avg = 10*log10(NMSE_asigw_avg/iteration);
NMSE_psigw_avg = 10*log10(NMSE_psigw_avg/iteration);
NMSE_sbl_near_avg = 10*log10(NMSE_sbl_near_avg/iteration);

figure;
xlabel('SNR (dB)');
ylabel('NMSE (dB)');
xlim(SNR_vec([1 length(SNR_vec)]));
%ylim([-25 0]);
grid on;
hold on;

plot(SNR_vec, NMSE_oracle_avg, "black--", 'DisplayName', 'Oracle-LS', 'MarkerSize', 3);
plot(SNR_vec, NMSE_ls_avg, "black<-", 'DisplayName', 'LS', 'MarkerSize', 3);
plot(SNR_vec, NMSE_asomp_avg, "cyan>--", 'DisplayName', 'Angular domain SOMP', 'MarkerSize', 3);
plot(SNR_vec, NMSE_psomp_avg, "cyan<-", 'DisplayName', 'Proposed P-SOMP', 'MarkerSize', 3);
plot(SNR_vec, NMSE_asigw_avg, "blue>--", 'DisplayName', 'Angular SIGW', 'MarkerSize', 3);
plot(SNR_vec, NMSE_psigw_avg, "blue<-", 'DisplayName', 'Proposed P-SIGW', 'MarkerSize', 3);
plot(SNR_vec, NMSE_sbl_near_avg, "redsquare-", 'DisplayName', 'Near field SBL', 'MarkerSize', 3);
 
legend('show');
hold off;





