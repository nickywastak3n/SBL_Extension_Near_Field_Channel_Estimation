% theta Derivative of Unit spatial signature (Near field)
function db_dtheta = der_theta_signature_near(N, theta, r, kc, d)
    delta = (2*(0:N-1)-N+1)/2;
    r_ant = sqrt(r^2+d^2*delta.^2-2*r*theta*d*delta);
    db_dtheta = signature_near(N, theta, r, kc, d) .* transpose((1i*kc*r*d*delta)./r_ant);
end