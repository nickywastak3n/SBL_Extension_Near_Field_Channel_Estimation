% Unit spatial signature (Far field)
function a = signature_far(N, theta)
    a = (1/sqrt(N))*transpose(exp(1i*pi*theta*(0:N-1))); % (N by 1)
end