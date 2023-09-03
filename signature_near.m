% Unit spatial signature (Near field)
function b = signature_near(N, theta, r, kc, d)
    delta = (2*(0:N-1)-N+1)/2;
    r_ant = sqrt(r^2+d^2*delta.^2-2*r*theta*d*delta);
    b = (1/sqrt(N))*transpose(exp((-1i*kc)*(r_ant-r))); % (N by 1)
end