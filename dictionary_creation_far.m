% Far Field Dictionary Creation
function F = dictionary_creation_far(N, len_col, theta)
    F = zeros(N, len_col);
    for col=1:len_col
        F(:,col) = signature_far(N, theta(col));
    end
end