function m = median_complex(data, dim)
%MEDIAN_COMPLEX Median for complex numbers
%------------------------------------------------------------------------------

if nargin<2 || isempty(dim)
    dim = 1;
end
m = median(real(data),dim) + 1i*median(imag(data),dim);

end