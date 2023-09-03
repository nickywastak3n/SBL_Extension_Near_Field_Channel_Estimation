% Derivative of Unit spatial signature (Far field)
function da_dtheta = der_theta_signature_far(N, theta)
    da_dtheta = signature_far(N, theta).*transpose(1i*pi*(0:N-1));
end