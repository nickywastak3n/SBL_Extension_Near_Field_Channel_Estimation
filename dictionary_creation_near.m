% Near Field Dictionary Creation
function W = dictionary_creation_near(N, len_col, theta, r, kc, d)
    W = zeros(N, len_col);
    for col=1:len_col
        W(:,col) = signature_near(N, theta(col), r(col), kc, d);
    end
end