% 1/r Derivative of Unit spatial signature (Near field)
function db_dr = der_r_signature_near(N, theta, r, kc, d)
    delta = (2*(0:N-1)-N+1)/2;
    r_ant = sqrt(r^2+d^2*delta.^2-2*r*theta*d*delta);
    db_dr = signature_near(N, theta, r, kc, d) .* transpose((-1i*kc)*(r^2+(-r^3+theta*d*r^2*delta)./r_ant));
end